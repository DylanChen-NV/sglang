# SPDX-License-Identifier: Apache-2.0
"""Native hybrid-cache extension of the replicated-weight CP executor.

Only request working/track slots are written here. The scheduler and native
radix cache retain all ownership, COW, publication, locking and eviction.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _affine_output_kernel(U, State, Out, E: tl.constexpr,
                          BM: tl.constexpr = 32, BN: tl.constexpr = 64):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.program_id(2)
    k = tl.arange(0, 32)
    hi = tl.full((BM, BN), 0, tl.float32)
    lo = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 128, 32):
        u = tl.load(U + (m[:, None] * 32 + h) * 128 + start + k[None, :],
                    m[:, None] < E, other=0)
        state = tl.load(State + (h * 128 + n[None, :]) * 128 + start + k[:, None])
        upper = state.to(tl.bfloat16)
        lower = (state - upper.to(tl.float32)).to(tl.bfloat16)
        hi = tl.dot(u, upper, hi)
        lo = tl.dot(u, lower, lo)
    pointer = Out + (m[:, None] * 32 + h) * 128 + n[None, :]
    original = tl.load(pointer, m[:, None] < E, other=0).to(tl.float32)
    tl.store(pointer, original + (hi + lo), m[:, None] < E)


@triton.jit
def _bf16_correction_kernel(U, State, Correction, Local, Final, E: tl.constexpr,
                            SPLIT_STATE: tl.constexpr,
                            FUSE_FINAL: tl.constexpr = False,
                            BM: tl.constexpr = 32, BN: tl.constexpr = 64):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.program_id(2)
    k = tl.arange(0, 32)
    hi = tl.full((BM, BN), 0, tl.float32)
    lo = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 128, 32):
        u = tl.load(U + (m[:, None] * 32 + h) * 128 + start + k[None, :],
                    m[:, None] < E, other=0)
        state = tl.load(State + (h * 128 + n[None, :]) * 128 + start + k[:, None])
        upper = state.to(tl.bfloat16)
        hi = tl.dot(u, upper, hi)
        if SPLIT_STATE:
            lower = (state - upper.to(tl.float32)).to(tl.bfloat16)
            lo = tl.dot(u, lower, lo)
    value = hi
    if SPLIT_STATE:
        value = hi + lo
    offset = (m[:, None] * 32 + h) * 128 + n[None, :]
    if FUSE_FINAL:
        original = tl.load(Local + offset, m[:, None] < E, other=0).to(tl.float32)
        tl.store(Final + offset, original + value, m[:, None] < E)
    else:
        tl.store(Correction + offset, value, m[:, None] < E)


def _bf16_correction(u, previous, split_state=False):
    correction = torch.empty(u.shape, dtype=torch.float32, device=u.device)
    _bf16_correction_kernel[(triton.cdiv(len(u), 32), 2, 32)](
        u, previous, correction, correction, correction, len(u), split_state, num_warps=4)
    return correction


def _affine_output_bf16_fused(out, u, previous, split_state=False):
    """Keep correction in FP32 registers, add O_local32, then store BF16."""
    if (out.dtype != torch.float32 or u.dtype != torch.bfloat16
            or previous.dtype != torch.float32 or out.shape != u.shape
            or tuple(u.shape[1:]) != (32,128) or tuple(previous.shape) != (1,32,128,128)
            or not all(t.is_contiguous() for t in (out,u,previous))):
        raise ValueError("Fused BF16 affine requires contiguous FP32 local/state and BF16 U")
    final = torch.empty(u.shape,dtype=torch.bfloat16,device=u.device)
    _bf16_correction_kernel[(triton.cdiv(len(u),32),2,32)](
        u,previous,final,out,final,len(u),split_state,True,num_warps=4)
    return final


@triton.jit
def _fp32_affine_epilogue_kernel(Out, Correction, E: tl.constexpr,
                                 BLOCK: tl.constexpr):
    """Add the completed cuBLAS dot, then perform the original BF16 rounding."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < E * 32 * 128
    token = index // (32 * 128)
    head = (index // 128) % 32
    dim = index % 128
    # torch.bmm writes contiguous [head, token, value], whereas Out is
    # contiguous [token, head, value]. The transpose is only a view.
    correction_index = (head * E + token) * 128 + dim
    original = tl.load(Out + index, mask, other=0).to(tl.float32)
    correction = tl.load(Correction + correction_index, mask, other=0)
    tl.store(Out + index, original + correction, mask)


class FP32AffineWorkspace:
    """Per-backend scratch, never a prefix state or a graph/stream-global cache.

    The CP executor serializes single-request forwards. Views use the current
    token count, including the cuBLAS batch stride, rather than the capacity.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.capacity = 0
        self.u32 = self.c32 = None

    def reserve(self, rows):
        if rows > self.capacity:
            count = rows * 32 * 128
            self.u32 = torch.empty(count, dtype=torch.float32, device=self.device)
            self.c32 = torch.empty(count, dtype=torch.float32, device=self.device)
            self.capacity = rows

    def u_view(self, rows):
        return self.u32[:rows * 32 * 128].view(rows, 32, 128)

    def c_view(self, rows):
        return self.c32[:rows * 32 * 128].view(32, rows, 128)


def _affine_output(out, u, previous, mode="two-bf16", fuse_output=False,
                   fp32_optimization="none", workspace=None):
    """Repair the second CP shard using the selected arithmetic policy.

    The low part retains the state bits lost by the original single BF16
    cast. Fuse both dot products and the final add to avoid materializing
    full-sequence FP32 input/output temporaries; persistent state stays FP32.
    """
    fp32_local = mode in ("bf16-fp32-local", "two-bf16-fp32-local")
    if fp32_optimization not in ("none", "epilogue", "workspace", "u-output"):
        raise ValueError(f"Unknown FP32 affine optimization: {fp32_optimization}")
    if fp32_optimization != "none" and mode != "fp32":
        raise ValueError("FP32 affine optimization requires fp32 arithmetic")
    if fuse_output and not fp32_local:
        raise ValueError("Fused CP affine output requires FP32 O_local")
    expected_out_dtype = torch.float32 if fp32_local else torch.bfloat16
    expected_u_dtype = torch.float32 if fp32_optimization == "u-output" else torch.bfloat16
    if (out.shape != u.shape or tuple(u.shape[1:]) != (32, 128)
            or tuple(previous.shape) != (1, 32, 128, 128)
            or out.dtype != expected_out_dtype or u.dtype != expected_u_dtype
            or previous.dtype != torch.float32
            or not all(t.is_contiguous() for t in (out, u, previous))):
        raise ValueError("CP affine correction requires contiguous output/U with the selected dtypes and FP32 state")
    if mode in ("single-bf16", "bf16-fp32-local", "two-bf16-fp32-local"):
        if fuse_output:
            return _affine_output_bf16_fused(out,u,previous,
                split_state=(mode == "two-bf16-fp32-local"))
        correction = _bf16_correction(u, previous,
            split_state=(mode == "two-bf16-fp32-local"))
        # Complete the BF16 dot in FP32 before adding O_local. The FP32-local
        # modes receive the actual pre-rounding GDN output, not BF16.float().
        return (out.float() + correction).to(torch.bfloat16)
    if mode == "fp32":
        # Do not let a global TF32 setting silently reduce this control's
        # mantissa precision. Both operands, the dot and the add stay FP32;
        # only the completed result is rounded back to the model dtype.
        if torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("FP32 CP affine correction requires TF32 matmul disabled")
        if fp32_optimization in ("workspace", "u-output"):
            if workspace is None or workspace.device != out.device:
                raise ValueError("FP32 affine workspace must belong to this model device")
            workspace.reserve(len(u))
            if fp32_optimization == "u-output":
                # GDN has already written FP32(BF16(acc)), not raw acc32.
                u32 = u
            else:
                u32 = workspace.u_view(len(u))
                u32.copy_(u)
            correction = workspace.c_view(len(u))
            # Preserve the original noncontiguous input and contiguous output
            # layouts exactly, even after a preceding longer request.
            torch.bmm(u32.transpose(0, 1), previous[0].transpose(1, 2), out=correction)
            correction = correction.transpose(0, 1)
        else:
            correction = torch.bmm(
                u.transpose(0, 1).float(), previous[0].transpose(1, 2)
            ).transpose(0, 1)
        if fp32_optimization in ("epilogue", "workspace", "u-output"):
            _fp32_affine_epilogue_kernel[(triton.cdiv(out.numel(), 1024),)](
                out, correction, len(out), BLOCK=1024)
            return out
        return (out.float() + correction).to(out.dtype)
    if mode != "two-bf16":
        raise ValueError(f"Unknown CP affine mode: {mode}")
    _affine_output_kernel[(triton.cdiv(len(u), 32), 2, 32)](
        u, previous, out, len(u), num_warps=4)
    return out


def checkpoint_offset(prefix, length, track_seqlen, chunk_size):
    offset = (track_seqlen - prefix) // chunk_size * chunk_size
    if not 0 < offset <= length:
        raise ValueError("CP checkpoint does not lie inside the new-token interval")
    return offset


def _gather_heads(tensor, ctx):
    parts = [torch.empty_like(tensor), torch.empty_like(tensor)]
    dist.all_gather(parts, tensor.contiguous(), group=ctx.group)
    return parts


def initialize(ctx, args):
    ctx.affine_mode = args.qwen35_cp_affine_mode
    ctx.affine_fuse_output = args.qwen35_cp_affine_fuse_output
    ctx.fp32_optimization = args.qwen35_cp_fp32_optimization
    ctx.fp32_refinement = args.qwen35_cp_fp32_refinement
    ctx.fp32_workspace = None
    if ctx.fp32_optimization in ("workspace", "u-output") and not ctx.single and ctx.rank == 1:
        backend = ctx.backend
        if not hasattr(backend, "_qwen35_cp_fp32_workspace"):
            backend._qwen35_cp_fp32_workspace = FP32AffineWorkspace(ctx.loc.device)
        ctx.fp32_workspace = backend._qwen35_cp_fp32_workspace
        ctx.fp32_workspace.reserve(ctx.size)
    if args.disable_radix_cache or not args.enable_mamba_extra_buffer():
        raise ValueError("CP prefix reuse requires enabled radix cache and extra_buffer Mamba tracking")
    pool = ctx.backend.req_to_token_pool
    if pool.mamba_ckpt_pool is not None:
        raise ValueError("CP prefix reuse requires unquantized FP32 checkpoints")
    ctx.prefix = int(ctx.batch.extend_prefix_lens_cpu[0])
    ctx.total = ctx.prefix + ctx.length
    if ctx.prefix < 0 or ctx.total > args.context_length:
        raise ValueError("CP prefix/new-token interval exceeds context capacity")
    ctx.prefix_loc = pool.req_to_token[ctx.batch.req_pool_indices[0], :ctx.prefix].long()
    ctx.initial_states = None
    ctx.checkpoint_states, ctx.checkpoint_convs = [], []
    ctx.track = 0
    mask = ctx.batch.mamba_track_mask
    if mask is not None and bool(mask[0].item()):
        ctx.track = checkpoint_offset(ctx.prefix, ctx.length,
            int(ctx.batch.mamba_track_seqlens[0].item()), args.mamba_cache_chunk_size)
        ctx.track_owner = 0 if ctx.track <= ctx.rows[0][1] else 1
        ctx.track_slot = pool.translate_mamba_indices(ctx.batch.mamba_track_indices[:1])
        cut = ctx.track - ctx.start if ctx.rank == ctx.track_owner else 0
        if 0 < cut < ctx.size:
            if cut % 64:
                raise ValueError("CP GDN checkpoint must align with the native 64-token kernel chunk")
            ctx.checkpoint_count = ctx.size // cut
            ctx.checkpoint_cu_starts = torch.tensor([0, ctx.checkpoint_count], dtype=torch.int32, device=ctx.loc.device)
    if ctx.prefix:
        mid = pool.translate_mamba_indices(pool.get_mamba_indices(ctx.batch.req_pool_indices))
        cache = pool.mamba_pool.mamba_cache
        # ModelRunner has completed deferred COW on this same forward stream.
        states = _gather_heads(cache.temporal[:, mid[0]].contiguous(), ctx)
        ctx.initial_states = torch.cat(states, dim=1)
        convs = _gather_heads(cache.conv[0][:, mid[0]].contiguous(), ctx)
        groups = [c.split((1024, 1024, 2048), dim=1) for c in convs]
        ctx.convs.copy_(torch.cat([torch.cat([groups[r][g] for r in (0, 1)], dim=1)
                                   for g in range(3)], dim=1))
        ctx.has_initial.fill_(True)
        ctx.seq_len.fill_(ctx.total)
        ctx.cu_kv = torch.tensor([0, ctx.total], dtype=torch.int32, device=ctx.loc.device)


def _conv_shard(convs, rank):
    return torch.cat([g[:, rank*(g.shape[1]//2):(rank+1)*(g.shape[1]//2)]
                      for g in convs.split((2048, 2048, 4096), dim=1)], dim=1).contiguous()


def _checkpoint_conv(initial, mixed, count):
    """Only the last three raw inputs belong to the conv checkpoint."""
    if not 0 < count <= len(mixed):
        raise ValueError("conv checkpoint outside the local new-token interval")
    tail = mixed[count-3:count] if count >= 3 else torch.cat([initial[0].T, mixed[:count]])[-3:]
    return tail.T.contiguous()[None]


def persist_checkpoint(ctx):
    if not ctx.track:
        return
    if ctx.rank == ctx.track_owner:
        full = torch.cat(ctx.checkpoint_states)
        convs = torch.cat(ctx.checkpoint_convs)
        local = full[:, ctx.rank*16:(ctx.rank+1)*16]
        conv = _conv_shard(convs, ctx.rank)
        ctx.exchange(send=full[:, (1-ctx.rank)*16:(2-ctx.rank)*16].contiguous())
        ctx.exchange(send=_conv_shard(convs, 1-ctx.rank))
    else:
        local = torch.empty(30, 16, 128, 128, dtype=torch.float32, device=ctx.loc.device)
        conv = torch.empty(30, 4096, 3, dtype=torch.bfloat16, device=ctx.loc.device)
        ctx.exchange(recv=local)
        ctx.exchange(recv=conv)
    cache = ctx.backend.req_to_token_pool.mamba_pool.mamba_cache
    cache.temporal[:, ctx.track_slot] = local[:, None]
    cache.conv[0][:, ctx.track_slot] = conv[:, None]
    ctx.checkpoint_states.clear()
    ctx.checkpoint_convs.clear()


def attention(layer, x, positions, ctx):
    """Ulysses for new queries, attending to cached prefix plus new keys."""
    import flashinfer
    from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
    from sglang.srt.models.qwen35_cp import copy_rows
    from sglang.srt.runtime_context import get_server_args

    w, r, E, L, P = layer.weights, ctx.rank, ctx.size, ctx.length, ctx.prefix
    bf = dict(dtype=torch.bfloat16, device=ctx.loc.device)
    s, e = ctx.start, ctx.end
    ps, pe = ctx.rows[1-r]
    qa = torch.empty(L, 8, 256, **bf)
    send = torch.empty(E, 2560, **bf)
    if E:
        qg, k, v = F.linear(x, w['qkv']).split([8192, 512, 512], -1)
        q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(qg, k, w['qn'], w['kn'],
            ctx.rope, positions, 1e-6, 16, 2, 256, 64, has_gate=True)
        q, k, v = q.view(E, 2, 8, 256), k.view(E, 2, 256), v.view(E, 2, 256)
        copy_rows(qa[s:e], q[:, r])
        copy_rows(ctx.kh[P+s:P+e, 0], k[:, r])
        copy_rows(ctx.vh[P+s:P+e, 0], v[:, r])
        copy_rows(send[:, :2048].view(E, 8, 256), q[:, 1-r])
        copy_rows(send[:, 2048:2304], k[:, 1-r])
        copy_rows(send[:, 2304:], v[:, 1-r])
    recv = torch.empty(pe-ps, 2560, **bf)
    ctx.exchange(send, recv)
    if pe > ps:
        copy_rows(qa[ps:pe], recv[:, :2048].view(-1, 8, 256))
        copy_rows(ctx.kh[P+ps:P+pe, 0], recv[:, 2048:2304])
        copy_rows(ctx.vh[P+ps:P+pe, 0], recv[:, 2304:])
    pool = ctx.backend.token_to_kv_pool
    ctx.kh[:P].copy_(pool.get_key_buffer(layer.layer_id)[ctx.prefix_loc])
    ctx.vh[:P].copy_(pool.get_value_buffer(layer.layer_id)[ctx.prefix_loc])
    pool.set_kv_buffer(layer, ctx.loc, ctx.kh[P:P+L], ctx.vh[P:P+L])
    kc = ctx.kh.view(-1, 64, 1, 256).permute(0, 2, 1, 3)
    vc = ctx.vh.view(-1, 64, 1, 256).permute(0, 2, 1, 3)
    out = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=qa, kv_cache=(kc, vc), workspace_buffer=ctx.workspace,
        block_tables=ctx.page_table, seq_lens=ctx.seq_len, max_q_len=L,
        max_kv_len=get_server_args().context_length, bmm1_scale=256**-0.5,
        bmm2_scale=1.0, batch_size=1, cum_seq_lens_q=ctx.cu_full,
        cum_seq_lens_kv=ctx.cu_kv, window_left=-1, out_dtype=torch.bfloat16)
    combined = torch.empty(E, 2, 8, 256, **bf)
    copy_rows(combined[:, r], out[s:e])
    recv = torch.empty(E, 8, 256, **bf)
    ctx.exchange(out[ps:pe], recv)
    copy_rows(combined[:, 1-r], recv)
    if not E:
        return x
    return F.linear(fused_sigmoid_mul(combined.reshape(E, 4096), gate, inplace=True), w['o'])


def gdn(layer, x, ctx):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
    from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

    w, E, r = layer.weights, ctx.size, ctx.rank
    if not E:
        return x
    conv = ctx.convs[layer.linear_id:layer.linear_id+1]
    if ctx.single:
        qkvz = F.linear(x, layer._qkvz)
        mixed, z = qkvz[:, :8192].contiguous(), qkvz[:, 8192:].reshape(E, 32, 128)
    else:
        mixed, z = F.linear(x, w['qkvw']), F.linear(x, w['zw']).view(E, 32, 128)
    ba = F.linear(x, w['ba'])
    b, a = ba[:, :32].contiguous(), ba[:, 32:].contiguous()
    if not ctx.single:
        halo = torch.empty(3, 8192, dtype=mixed.dtype, device=x.device)
        ctx.exchange(mixed[-3:].contiguous() if r == 0 else None, halo if r == 1 else None)
        if r == 1:
            conv[0].copy_(halo.T)
    cut = ctx.track - ctx.start if ctx.track and ctx.track_owner == r else 0
    checkpoint_conv = None
    if cut:
        checkpoint_conv = _checkpoint_conv(conv, mixed, cut)
    xc = causal_conv1d_fn(mixed.T, w['conv'], None, conv_states=conv,
        query_start_loc=ctx.cu_local, seq_lens_cpu=[E], cache_indices=ctx.zero_slot,
        has_initial_state=ctx.has_initial, activation='silu').T[:E]
    q, k, v = fused_qkv_split_gdn_prefill(xc, 16, 16, 32, 128, 128, 128)
    g, beta = fused_gdn_gating(w['A_log'], a, b, w['dt_bias'])
    q, k, v = l2norm_fwd(q[0].contiguous()), l2norm_fwd(k[0].contiguous()), v[0].contiguous()
    alpha, beta = torch.exp(g[0].float()), beta[0].float()

    def run(values, initial, fp32_output=False, rounded_output=False):
        options = {}
        checkpoints = None
        if 0 < cut < E:
            checkpoints = torch.empty(ctx.checkpoint_count,32,128,128,dtype=torch.float32,device=x.device)
            options = dict(state_checkpoints=checkpoints, checkpoint_cu_starts=ctx.checkpoint_cu_starts,
                           checkpoint_every_n_tokens=cut)
        kernel = chunk_gated_delta_rule
        if fp32_output or rounded_output:
            from sglang.kernels.ops.attention.qwen35_gdn_fp32 import chunk_gated_delta_rule_fp32_output
            kernel = chunk_gated_delta_rule_fp32_output
            if rounded_output:
                options.update(output_fp32_buffer=ctx.fp32_workspace.u_view(E),
                               round_output_to_bf16=True,
                               emit_bf16_output=(ctx.fp32_refinement == "none"))
        out,state = kernel(q=q, k=k, v=values, g=alpha, beta=beta,
            scale=None, initial_state=initial, output_final_state=True, cu_seqlens=ctx.cu_local,
            use_qk_l2norm_in_kernel=False, output_state=torch.empty_like(initial), **options)
        checkpoint = checkpoints[:1].clone() if checkpoints is not None else (state if cut == E else None)
        return out,state,checkpoint

    initial = ctx.initial_states[layer.linear_id:layer.linear_id+1] if ctx.prefix and r == 0 else ctx.zero_state
    fp32_local = (not ctx.single and r == 1
                  and ctx.affine_mode in ("bf16-fp32-local", "two-bf16-fp32-local"))
    out, state, checkpoint = run(v, initial, fp32_output=fp32_local)
    if not ctx.single:
        if r == 0:
            ctx.exchange(send=state)
        else:
            u, transition, checkpoint_transition = run(ctx.zero_v[:E], ctx.eye,
                rounded_output=(ctx.fp32_optimization == "u-output"))
            previous = torch.empty_like(state)
            ctx.exchange(recv=previous)
            # Retain the FP32 recurrent state during the affine output repair.
            # Rounding the restored state to BF16 here compounds prefix/cold
            # differences through the subsequent MoE layers.
            out = _affine_output(out, u, previous, mode=ctx.affine_mode,
                                 fuse_output=ctx.affine_fuse_output,
                                 fp32_optimization=ctx.fp32_optimization,
                                 workspace=ctx.fp32_workspace)
            state = (torch.bmm(previous[0].float(), transition[0].float()) + state[0].float())[None]
            if cut:
                checkpoint = (torch.bmm(previous[0].float(), checkpoint_transition[0].float()) + checkpoint[0].float())[None]
    if r == (0 if ctx.single else 1):
        ctx.final_states.append(state)
    if cut:
        ctx.checkpoint_states.append(checkpoint)
        ctx.checkpoint_convs.append(checkpoint_conv)
    y = layernorm_fn(out.reshape(-1, 128), w['gnorm'], None, z=z.reshape(-1, 128),
        eps=1e-6, group_size=None, norm_before_gate=True, is_rms_norm=True, activation='swish')
    return F.linear(y.view(E, 4096), w['out'])
