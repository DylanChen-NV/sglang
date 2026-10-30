# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 cold-prefill CP layers.

Algorithm reference: AliceChenyy/qwen35-1p1d@d4cfed0,
q35rt/prefill/{engine,cp}.py. Keep the reference operations and precision;
SGLang owns request allocation, persistent pools, sampling and scheduling.
Native prefix reuse is opt-in through the CP prefix-cache extension.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def enabled():
    from sglang.srt.runtime_context import get_server_args

    return getattr(get_server_args(), "enable_qwen35_cp_prefill", False)


def uses_reference_kernels(batch):
    """Cold CP owns kernel metadata; SGLang still owns the persistent pools."""
    return (
        enabled()
        and batch.forward_mode.name == "EXTEND"
        and batch.batch_size == 1
        and len(batch.extend_prefix_lens_cpu) == 1
        and batch.extend_prefix_lens_cpu[0] >= 0
        and not batch.return_logprob
    )


def split_rows(length, ratio=0.48, minimum=4096, align_cut=False):
    if length < 1 or not 0 < ratio < 1:
        raise ValueError("invalid CP length or ratio")
    if length < minimum:
        return ((0, length), (length, length))
    n1 = int(length * ratio) // 64 * 64
    cut = length - n1
    if align_cut:
        cut = cut // 64 * 64
    rows = ((0, cut), (cut, length))
    if max(b - a for a, b in rows) > 49152:
        raise ValueError("CP per-rank rows exceed the reference 49152 limit")
    return rows


def copy_rows(dst, src):
    dst.view(torch.int64).copy_(src.view(torch.int64))


@dataclass
class CPContext:
    length: int
    rank: int
    group: object
    rows: tuple
    loc: torch.Tensor
    batch: object
    workspace: torch.Tensor
    rope: torch.Tensor
    kh: torch.Tensor
    vh: torch.Tensor
    page_table: torch.Tensor
    eye: torch.Tensor
    zero_state: torch.Tensor
    zero_v: torch.Tensor
    convs: torch.Tensor
    zero_slot: torch.Tensor
    pynccl_comm: object = None
    diagnostic_root: str = None

    @property
    def start(self):
        return self.rows[self.rank][0]

    @property
    def end(self):
        return self.rows[self.rank][1]

    @property
    def size(self):
        return self.end - self.start

    @property
    def single(self):
        return self.rows[1][0] == self.length

    @property
    def backend(self):
        from sglang.srt.model_executor.forward_context import get_attn_backend
        return get_attn_backend()

    def exchange(self, send=None, recv=None):
        has_send = send is not None and send.numel() > 0
        has_recv = recv is not None and recv.numel() > 0
        if not has_send and not has_recv:
            return
        if self.pynccl_comm is not None:
            # The supported executor has one active request, no overlap
            # scheduler, and allocates/consumes these buffers on this stream.
            # No allocator lifetime crosses to a separate communication stream.
            comm = self.pynccl_comm
            with comm.change_state(enable=True):
                comm.group_start()
                try:
                    if has_send:
                        comm.send(send, 1 - self.rank)
                    if has_recv:
                        comm.recv(recv, 1 - self.rank)
                finally:
                    comm.group_end()
            return
        peer = dist.get_global_rank(self.group, 1 - self.rank)
        ops = []
        if send is not None and send.numel():
            ops.append(dist.P2POp(dist.isend, send, peer, self.group))
        if recv is not None and recv.numel():
            ops.append(dist.P2POp(dist.irecv, recv, peer, self.group))
        for work in dist.batch_isend_irecv(ops):
            work.wait()

    def share_from(self, tensor, source):
        dist.broadcast(tensor, src=dist.get_global_rank(self.group, source), group=self.group)
        return tensor


def rope_cache(dim, max_pos, device):
    inv = 1.0 / (1e7 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    f = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=device), inv)
    return torch.cat([f.cos(), f.sin()], -1).contiguous()


def attention_metadata(ctx):
    return ctx.seq_len, ctx.cu_full, ctx.cu_full


def gdn_metadata(ctx):
    return ctx.cu_local, ctx.has_initial


def p2p_communicator(group):
    """Choose one backend collectively, once, before the first model forward.

    Use SGLang's existing communicator; never fall back after a NCCL error.
    Its send/recv API submits on the caller's current stream.
    """
    comm = group.pynccl_comm
    available = comm is not None and comm.available
    agreed = torch.tensor([int(available)], dtype=torch.int32, device="cpu")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if not agreed.item():
        return None
    if comm.world_size != 2 or comm.rank != group.rank_in_group:
        raise RuntimeError("CP communicator does not match the TP group")
    return comm


def make_context(model, input_ids, batch):
    from sglang.srt.runtime_context import get_parallel, get_server_args

    args = get_server_args()
    reuse = getattr(args, "enable_qwen35_cp_prefix_cache", False)
    if (not reuse and not args.disable_radix_cache) or args.chunked_prefill_size != -1:
        raise ValueError("M2 CP requires --disable-radix-cache --chunked-prefill-size -1")
    if args.mamba_ssm_dtype != "float32":
        raise ValueError("M2 CP requires --mamba-ssm-dtype float32 to retain reference state precision")
    if get_parallel().tp_size != 2 or get_parallel().pp_size != 1:
        raise ValueError("Qwen3.5 reference CP requires TP world 2 and PP 1")
    if batch.forward_mode.name != "EXTEND" or batch.batch_size != 1:
        raise ValueError("M2 CP supports one cold extend request only")
    if not reuse and batch.extend_prefix_lens_cpu != [0]:
        raise ValueError("M2 CP rejects prefix reuse")
    if batch.return_logprob:
        raise ValueError("M2 CP does not support prompt logprobs")
    length = input_ids.numel()
    if length != batch.extend_seq_lens_cpu[0]:
        raise ValueError("M2 CP requires a complete, unchunked prompt")
    dev = input_ids.device
    capacity = math.ceil(args.context_length / 64) * 64
    if not hasattr(model, "_cp_scratch"):
        model._cp_scratch = dict(
            workspace=torch.zeros(512 << 20, dtype=torch.uint8, device=dev),
            rope=rope_cache(64, capacity, dev),
            kh=torch.zeros(capacity, 1, 256, dtype=torch.bfloat16, device=dev),
            vh=torch.zeros(capacity, 1, 256, dtype=torch.bfloat16, device=dev),
            page_table=torch.arange(capacity // 64, dtype=torch.int32, device=dev)[None],
            eye=torch.eye(128, dtype=torch.float32, device=dev).expand(1, 32, 128, 128).contiguous(),
            zero_state=torch.zeros(1, 32, 128, 128, dtype=torch.float32, device=dev),
            zero_v=torch.zeros(capacity, 32, 128, dtype=torch.bfloat16, device=dev),
            convs=torch.zeros(30, 8192, 3, dtype=torch.bfloat16, device=dev),
            zero_slot=torch.zeros(1, dtype=torch.int32, device=dev),
            pynccl_comm=p2p_communicator(get_parallel().tp_group),
        )
    model._cp_scratch["convs"].zero_()
    ctx = CPContext(length, get_parallel().tp_rank,
                    get_parallel().tp_group.device_group,
                    split_rows(length, align_cut=reuse), batch.out_cache_loc.long(), batch,
                    **model._cp_scratch,
                    diagnostic_root=os.environ.get("Q35_CP_LAYER_DIAGNOSTICS"))
    batch.qwen35_cp_context = ctx
    ctx.final_states = []
    # These values are identical across the ten attention / thirty GDN layers.
    # Construct once before launching model work instead of repeating tiny
    # synchronous host-to-device transfers between layers.
    ctx.seq_len = torch.tensor([length], dtype=torch.int32, device=dev)
    ctx.cu_full = torch.tensor([0, length], dtype=torch.int32, device=dev)
    ctx.cu_local = torch.tensor([0, ctx.size], dtype=torch.int32, device=dev)
    ctx.has_initial = torch.tensor([not ctx.single and ctx.rank == 1], device=dev)
    ctx.prefix = 0
    ctx.reuse = reuse
    if reuse:
        from sglang.srt.models.qwen35_cp_cache import initialize
        initialize(ctx, args)
    return ctx


def persist_states(ctx):
    """Write native head shards after all reference layer computation.

    Only the missing half of the terminal state crosses ranks, once per
    request rather than adding a synchronization point to every GDN layer.
    """
    owner = 0 if ctx.single else 1
    rank, dev = ctx.rank, ctx.loc.device
    if rank == owner:
        states = torch.cat(ctx.final_states, dim=0)
        convs = ctx.convs
        def heads(r):
            return torch.cat([convs[:, :2048][:, r*1024:(r+1)*1024],
                              convs[:, 2048:4096][:, r*1024:(r+1)*1024],
                              convs[:, 4096:][:, r*2048:(r+1)*2048]], dim=1)
        local_state = states[:, rank*16:(rank+1)*16]
        local_conv = heads(rank)
        ctx.exchange(send=states[:, (1-rank)*16:(2-rank)*16].contiguous())
        ctx.exchange(send=heads(1-rank).contiguous())
    else:
        local_state = torch.empty(30, 16, 128, 128, dtype=torch.float32, device=dev)
        local_conv = torch.empty(30, 4096, 3, dtype=torch.bfloat16, device=dev)
        ctx.exchange(recv=local_state)
        ctx.exchange(recv=local_conv)
    pool = ctx.backend.req_to_token_pool
    mid = pool.translate_mamba_indices(pool.get_mamba_indices(ctx.batch.req_pool_indices))
    cache = pool.mamba_pool.mamba_cache
    cache.temporal[:, mid] = local_state[:, None]
    cache.conv[0][:, mid] = local_conv[:, None]
    ctx.final_states.clear()
    if ctx.reuse:
        from sglang.srt.models.qwen35_cp_cache import persist_checkpoint
        persist_checkpoint(ctx)


def write_layer_probe(ctx, layer, kind, value):
    """Opt-in sampled intermediate outputs; never enabled in formal timing."""
    if not ctx.diagnostic_root or not len(value):
        return
    size = len(value)
    indices = sorted({0, min(1, size-1), min(2, size-1), size-1,
                      max(0, size-2)} | {i*size//16 for i in range(16)})
    selected = value.index_select(0, torch.tensor(indices, device=value.device)).detach().cpu()
    root = Path(ctx.diagnostic_root)/f'L{ctx.length}'
    root.mkdir(parents=True, exist_ok=True)
    torch.save(dict(global_rows=[ctx.start+i for i in indices], output=selected),
               root/f'rank{ctx.rank}.layer{layer:02d}.{kind}.pt')


def param(*shape, dtype=None):
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class CPDecoderLayer(nn.Module):
    """One native model layer, with replicated weights and local token rows."""

    def __init__(self, config, layer_id):
        super().__init__()
        if (config.hidden_size, config.num_hidden_layers, config.num_experts) != (2048, 40, 256):
            raise ValueError("M2 CP is fixed to Qwen3.5-35B-A3B")
        self.layer_id = layer_id
        self.full = (layer_id + 1) % 4 == 0
        self.linear_id = layer_id - layer_id // 4
        shapes = dict(in_ln=(2048,), post_ln=(2048,), router=(256, 2048),
                      se_gu=(1024, 2048), se_down=(2048, 512), se_gate=(2048,),
                      w13=(256, 32, 1024, 64), w2=(256, 8, 2048, 64))
        if self.full:
            shapes.update(qkv=(9216, 2048), qn=(256,), kn=(256,), o=(2048, 4096))
        else:
            shapes.update(qkvw=(8192, 2048), zw=(4096, 2048), ba=(64, 2048),
                          conv=(8192, 4), gnorm=(128,), out=(2048, 4096), dt_bias=(32,))
        self.weights = nn.ParameterDict({k: param(*s) for k, s in shapes.items()})
        if not self.full:
            self.weights["A_log"] = param(32, dtype=torch.float32)

    def load_tensor(self, name, value):
        """Load checkpoint tensors once, without a second model/checkpoint reader."""
        if not hasattr(self, "_loaded_parts"):
            self._loaded_parts = set()
        self._loaded_parts.add(name)
        w = self.weights
        simple = {"input_layernorm.weight": "in_ln", "post_attention_layernorm.weight": "post_ln",
                  "mlp.gate.weight": "router", "mlp.shared_expert.down_proj.weight": "se_down",
                  "mlp.shared_expert_gate.weight": "se_gate",
                  "self_attn.q_norm.weight": "qn", "self_attn.k_norm.weight": "kn",
                  "self_attn.o_proj.weight": "o", "linear_attn.in_proj_qkv.weight": "qkvw",
                  "linear_attn.in_proj_z.weight": "zw", "linear_attn.conv1d.weight": "conv",
                  "linear_attn.A_log": "A_log", "linear_attn.dt_bias": "dt_bias",
                  "linear_attn.norm.weight": "gnorm", "linear_attn.out_proj.weight": "out"}
        if name in simple:
            key = simple[name]
            w[key].data.copy_(value.reshape(w[key].shape))
            return key
        pieces = {"self_attn.q_proj.weight": ("qkv", 0, 8192),
                  "self_attn.k_proj.weight": ("qkv", 8192, 8704),
                  "self_attn.v_proj.weight": ("qkv", 8704, 9216),
                  "linear_attn.in_proj_b.weight": ("ba", 0, 32),
                  "linear_attn.in_proj_a.weight": ("ba", 32, 64),
                  "mlp.shared_expert.gate_proj.weight": ("se_gu", 0, 512),
                  "mlp.shared_expert.up_proj.weight": ("se_gu", 512, 1024)}
        if name in pieces:
            key, a, b = pieces[name]
            w[key].data[a:b].copy_(value)
            return key
        if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
            from flashinfer.fused_moe.core import (_maybe_get_cached_w3_w1_permute_indices,
                                                  get_w2_permute_indices_with_cache)
            if name.endswith("gate_up_proj"):
                gate, up = value.chunk(2, dim=-2)
                raw = torch.cat([up, gate], dim=1).to(w["w13"].device).view(torch.uint8)
                indices = _maybe_get_cached_w3_w1_permute_indices({}, raw[0], 128, is_gated_act_gemm=True)
                key = "w13"
            else:
                raw = value.to(w["w2"].device).view(torch.uint8)
                indices = get_w2_permute_indices_with_cache({}, raw[0], 128)
                key = "w2"
            shuffled = raw[:, indices.to(raw.device)]
            packed = shuffled.view(shuffled.shape[0], shuffled.shape[1], -1, 128).permute(0, 2, 1, 3).contiguous().view(torch.bfloat16)
            w[key].data.copy_(packed)
            return key
        return None

    def validate_loaded(self):
        common = {"input_layernorm.weight", "post_attention_layernorm.weight", "mlp.gate.weight",
                  "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
                  "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight",
                  "mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
        if self.full:
            names = {f"self_attn.{key}.weight" for key in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj")}
        else:
            names = {f"linear_attn.{key}.weight" for key in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "conv1d", "norm", "out_proj")}
            names.update(("linear_attn.A_log", "linear_attn.dt_bias"))
        missing = (common | names) - getattr(self, "_loaded_parts", set())
        if missing:
            raise ValueError(f"CP layer {self.layer_id}: missing checkpoint tensors {sorted(missing)}")
        if not self.full:
            # The reference short-input fallback uses one QKVZ GEMM. Keep its
            # contiguous backing storage and expose views for the CP GEMMs.
            self.register_buffer("_qkvz", torch.cat([self.weights["qkvw"], self.weights["zw"]]))
            self.weights["qkvw"].data = self._qkvz[:8192]
            self.weights["zw"].data = self._qkvz[8192:]

    def moe(self, x):
        import sgl_kernel as sk
        from flashinfer.fused_moe import trtllm_bf16_moe
        from flashinfer.fused_moe.core import ActivationType
        from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add

        w = self.weights
        shared = F.linear(sk.silu_and_mul(F.linear(x, w["se_gu"])), w["se_down"])
        out = trtllm_bf16_moe(
            routing_logits=F.linear(x, w["router"]), routing_bias=None, hidden_states=x,
            gemm1_weights=w["w13"], gemm2_weights=w["w2"], num_experts=256,
            top_k=8, n_group=None, topk_group=None, intermediate_size=512,
            local_expert_offset=0, local_num_experts=256, routed_scaling_factor=None,
            routing_method_type=4, tune_max_num_tokens=1 << (len(x) - 1).bit_length(),
            activation_type=ActivationType.Swiglu.value)
        fused_gate_sigmoid_mul_add(x, w["se_gate"], shared, out)
        return out

    def attention(self, x, positions, ctx):
        if ctx.prefix:
            from sglang.srt.models.qwen35_cp_cache import attention
            return attention(self, x, positions, ctx)
        import flashinfer
        from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import fused_qk_gemma_rmsnorm_rope_gate
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
        from sglang.srt.runtime_context import get_server_args

        w, r, E, L = self.weights, ctx.rank, ctx.size, ctx.length
        bf = dict(dtype=torch.bfloat16, device=ctx.loc.device)
        if E:
            qg, k, v = F.linear(x, w["qkv"]).split([8192, 512, 512], -1)
            q, k, gate = fused_qk_gemma_rmsnorm_rope_gate(
                qg, k, w["qn"], w["kn"], ctx.rope, positions, 1e-6,
                16, 2, 256, 64, has_gate=True)
            q, k, v = q.view(E, 2, 8, 256), k.view(E, 2, 256), v.view(E, 2, 256)
        if ctx.single:
            if r == 0:
                local_k, local_v = k[:, :1].contiguous(), v[:, :1].contiguous()
                send = torch.stack([k[:, 1].contiguous(), v[:, 1].contiguous()])
                ctx.exchange(send=send)
            else:
                recv = torch.empty(2, L, 256, **bf)
                ctx.exchange(recv=recv)
                local_k, local_v = recv[0, :, None], recv[1, :, None]
        else:
            s, e = ctx.start, ctx.end
            ps, pe = ctx.rows[1 - r]
            qa = torch.empty(L, 8, 256, **bf)
            copy_rows(qa[s:e], q[:, r])
            copy_rows(ctx.kh[s:e, 0], k[:, r])
            copy_rows(ctx.vh[s:e, 0], v[:, r])
            send = torch.empty(E, 2560, **bf)
            copy_rows(send[:, :2048].view(E, 8, 256), q[:, 1-r])
            copy_rows(send[:, 2048:2304], k[:, 1-r])
            copy_rows(send[:, 2304:], v[:, 1-r])
            recv = torch.empty(pe-ps, 2560, **bf)
            ctx.exchange(send, recv)
            copy_rows(qa[ps:pe], recv[:, :2048].view(-1, 8, 256))
            copy_rows(ctx.kh[ps:pe, 0], recv[:, 2048:2304])
            copy_rows(ctx.vh[ps:pe, 0], recv[:, 2304:])
            local_k, local_v = ctx.kh[:L], ctx.vh[:L]
        pool = ctx.backend.token_to_kv_pool
        # Keep pool-managed physical-slot translation and fuse the K/V store.
        pool.set_kv_buffer(self, ctx.loc, local_k, local_v)
        if not E:
            return x
        if ctx.single:
            # Reference single-GPU fallback uses both KV heads and all query heads.
            kk = torch.zeros_like(ctx.kh).expand(-1, 2, -1).contiguous()
            vv = torch.zeros_like(kk)
            kk[:L], vv[:L] = k, v
            query = q.reshape(L, 16, 256)
        else:
            kk, vv, query = ctx.kh, ctx.vh, qa
        heads = kk.shape[1]
        kc = kk.view(-1, 64, heads, 256).permute(0, 2, 1, 3)
        vc = vv.view(-1, 64, heads, 256).permute(0, 2, 1, 3)
        seq_len, cu_q, cu_kv = attention_metadata(ctx)
        out = flashinfer.prefill.trtllm_batch_context_with_kv_cache(
            query=query, kv_cache=(kc, vc), workspace_buffer=ctx.workspace,
            block_tables=ctx.page_table, seq_lens=seq_len, max_q_len=L,
            max_kv_len=get_server_args().context_length, bmm1_scale=256 ** -0.5,
            bmm2_scale=1.0, batch_size=1, cum_seq_lens_q=cu_q,
            cum_seq_lens_kv=cu_kv, window_left=-1, out_dtype=torch.bfloat16)
        if not ctx.single:
            combined = torch.empty(E, 2, 8, 256, **bf)
            copy_rows(combined[:, r], out[s:e])
            recv = torch.empty(E, 8, 256, **bf)
            ctx.exchange(out[ps:pe], recv)
            copy_rows(combined[:, 1-r], recv)
            out = combined
        out = fused_sigmoid_mul(out.reshape(E, 4096), gate, inplace=True)
        return F.linear(out, w["o"])

    def gdn(self, x, ctx):
        if ctx.reuse:
            from sglang.srt.models.qwen35_cp_cache import gdn
            return gdn(self, x, ctx)
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        from sglang.kernels.ops.attention.fla.layernorm_gated import layernorm_fn
        from sglang.kernels.ops.attention.triton_gdn_fused_proj import fused_qkv_split_gdn_prefill
        from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn

        w, E, r = self.weights, ctx.size, ctx.rank
        dev = ctx.loc.device
        conv = ctx.convs[self.linear_id:self.linear_id+1]
        if E:
            if ctx.single:
                qkvz = F.linear(x, self._qkvz)
                mixed = qkvz[:, :8192].contiguous()
                z = qkvz[:, 8192:].reshape(E, 32, 128)
            else:
                mixed = F.linear(x, w["qkvw"])
                z = F.linear(x, w["zw"]).view(E, 32, 128)
            ba = F.linear(x, w["ba"])
            b, a = ba[:, :32].contiguous(), ba[:, 32:].contiguous()
            if not ctx.single:
                halo = torch.empty(3, 8192, dtype=mixed.dtype, device=dev)
                ctx.exchange(mixed[-3:].contiguous() if r == 0 else None, halo if r == 1 else None)
                if r == 1:
                    conv[0].copy_(halo.T)
            cu, has_initial = gdn_metadata(ctx)
            xc = causal_conv1d_fn(
                mixed.T, w["conv"], None, conv_states=conv,
                query_start_loc=cu, seq_lens_cpu=[E],
                cache_indices=ctx.zero_slot,
                has_initial_state=has_initial,
                activation="silu").T[:E]
            q, k, v = fused_qkv_split_gdn_prefill(xc, 16, 16, 32, 128, 128, 128)
            g, beta = fused_gdn_gating(w["A_log"], a, b, w["dt_bias"])
            q, k, v = l2norm_fwd(q[0].contiguous()), l2norm_fwd(k[0].contiguous()), v[0].contiguous()
            alpha, beta = torch.exp(g[0].float()), beta[0].float()
            def run(values, initial):
                return chunk_gated_delta_rule(
                    q=q, k=k, v=values, g=alpha, beta=beta,
                    scale=None, initial_state=initial, output_final_state=True,
                    cu_seqlens=cu, use_qk_l2norm_in_kernel=False,
                    output_state=torch.empty_like(initial))
            out, state = run(v, ctx.zero_state)
            if not ctx.single:
                if r == 0:
                    ctx.exchange(send=state)
                else:
                    u, transition = run(ctx.zero_v[:E], ctx.eye)
                    previous = torch.empty_like(state)
                    ctx.exchange(recv=previous)
                    out.transpose(0, 1).baddbmm_(u.transpose(0, 1), previous[0].to(u.dtype).transpose(1, 2))
                    state = (torch.bmm(previous[0].float(), transition[0].float()) + state[0].float())[None]
        owner = 0 if ctx.single else 1
        if r == owner:
            ctx.final_states.append(state)
        if not E:
            return x
        y = layernorm_fn(out.reshape(-1, 128), w["gnorm"], None, z=z.reshape(-1, 128),
                         eps=1e-6, group_size=None, norm_before_gate=True,
                         is_rms_norm=True, activation="swish")
        return F.linear(y.view(E, 4096), w["out"])

    def forward(self, positions, hidden_states, residual, forward_batch,
                captured_last_layer_outputs=None, **kwargs):
        import sgl_kernel as sk

        ctx = forward_batch.qwen35_cp_context
        w = self.weights
        if ctx.size:
            if residual is None:
                residual, x = hidden_states, sk.gemma_rmsnorm(hidden_states, w["in_ln"], 1e-6)
            else:
                sk.gemma_fused_add_rmsnorm(hidden_states, residual, w["in_ln"], 1e-6)
                x = hidden_states
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.append(residual.clone())
        else:
            x = hidden_states
            if captured_last_layer_outputs is not None:
                captured_last_layer_outputs.append(hidden_states)
        a = self.attention(x, positions, ctx) if self.full else self.gdn(x, ctx)
        if ctx.size:
            write_layer_probe(ctx, self.layer_id, "attention" if self.full else "gdn", a)
            sk.gemma_fused_add_rmsnorm(a, residual, w["post_ln"], 1e-6)
            hidden_states = self.moe(a)
            write_layer_probe(ctx, self.layer_id, "moe", hidden_states)
        return hidden_states, residual


def load_weights(model, weights):
    """Native checkpoint iterator -> replicated CP layers and native top-level weights."""
    from sglang.srt.model_loader.weight_utils import default_weight_loader

    body = model.model if hasattr(model, "model") else model
    loaded = set()
    for name, value in weights:
        name = name.replace("model.language_model.", "model.")
        body_name = name[6:] if name.startswith("model.") else name
        if body_name.startswith("layers."):
            _, number, local = body_name.split(".", 2)
            key = body.layers[int(number)].load_tensor(local, value)
            if key is not None:
                loaded.add(f"model.layers.{number}.weights.{key}")
        elif body_name in ("embed_tokens.weight", "norm.weight"):
            obj = body.embed_tokens.weight if body_name.startswith("embed") else body.norm.weight
            getattr(obj, "weight_loader", default_weight_loader)(obj, value)
            loaded.add("model." + body_name)
        elif name == "lm_head.weight" and hasattr(model, "lm_head"):
            obj = model.lm_head.weight
            getattr(obj, "weight_loader", default_weight_loader)(obj, value)
            loaded.add(name)
    for layer in body.layers:
        layer.validate_loaded()
    return loaded


def model_forward(model, input_ids, positions, batch, input_embeds=None):
    import sgl_kernel as sk

    ctx = make_context(model, input_ids, batch)
    positions = torch.arange(ctx.prefix + ctx.start, ctx.prefix + ctx.end, dtype=torch.int64, device=input_ids.device)
    hidden = model.embed_tokens(input_ids[ctx.start:ctx.end]) if input_embeds is None else input_embeds[ctx.start:ctx.end]
    residual, aux = None, []
    for layer in model.layers:
        hidden, residual = layer(positions, hidden, residual, batch,
                                 captured_last_layer_outputs=aux if getattr(layer, "_is_layer_to_capture", False) else None)
    persist_states(ctx)
    last = torch.empty(1, 2048, dtype=torch.bfloat16, device=input_ids.device)
    owner = 0 if ctx.single else 1
    if ctx.rank == owner:
        sk.gemma_fused_add_rmsnorm(hidden[-1:], residual[-1:], model.norm.weight, 1e-6)
        last.copy_(hidden[-1:])
    ctx.share_from(last, owner)
    return last, aux


def logits_output(model, input_ids, positions, batch, input_embeds=None):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput

    last, aux = model.model(input_ids, positions, batch, input_embeds)
    logits = torch.empty(1, model.config.vocab_size, dtype=torch.bfloat16, device=last.device)
    ctx = batch.qwen35_cp_context
    owner = 0 if ctx.single else 1
    if ctx.rank == owner:
        logits.copy_(F.linear(last, model.lm_head.weight)[:, :model.config.vocab_size])
    ctx.share_from(logits, owner)
    return LogitsProcessorOutput(next_token_logits=logits.float(),
                                 hidden_states=torch.cat(aux, -1) if aux else None)


class CPDraftWeights(nn.Module):
    """Replicated context-only K/V projections, loaded by the draft iterator."""

    def __init__(self):
        super().__init__()
        self.kv = param(6, 2048, 2048)
        self.kn = param(6, 128)

    def load_tensor(self, name, value):
        name = name.removeprefix("model.")
        if not name.startswith("layers."):
            return
        _, number, local = name.split(".", 2)
        number = int(number)
        if local == "self_attn.k_proj.weight":
            self.kv.data[number, :1024].copy_(value)
        elif local == "self_attn.v_proj.weight":
            self.kv.data[number, 1024:].copy_(value)
        elif local == "self_attn.k_norm.weight":
            self.kn.data[number].copy_(value)


def materialize_draft(worker, hidden, cache_loc, positions):
    import sgl_kernel as sk
    from sglang.kernels.ops.speculative.fused_kv_materialize import _fused_norm_rope_stacked
    from sglang.srt.runtime_context import get_parallel

    rank = get_parallel().tp_rank
    rows = split_rows(cache_loc.numel(), align_cut=getattr(worker.model_runner.server_args, "enable_qwen35_cp_prefix_cache", False))
    start, end = rows[rank]
    peer_start, peer_end = rows[1-rank]
    group = get_parallel().tp_group.device_group
    peer = dist.get_global_rank(group, 1-rank)
    draft = worker.draft_model
    weights = draft.cp_context_weights
    if not hasattr(weights, "projection"):
        weights.projection = weights.kv.flatten(0, 1).T.contiguous()
        weights.rope = rope_cache(128, math.ceil(worker.model_runner.server_args.context_length / 64) * 64, cache_loc.device)
        weights.eps = torch.full((6,), 1e-6, dtype=torch.float32, device=cache_loc.device)
    local_len = end-start
    bf = dict(dtype=torch.bfloat16, device=cache_loc.device)
    if local_len:
        context = sk.rmsnorm(F.linear(hidden, draft.fc.weight), draft.hidden_norm.weight, 1e-6)
        projected = torch.mm(context, weights.projection).view(local_len, 6, 2048)
        dk, dv = _fused_norm_rope_stacked(projected, weights.kn, weights.eps,
                                         weights.rope, positions[start:end], 8, 128, 128)
        # The native writer accepts a strided source row. Avoid copying the
        # local head shard into another packed tensor just to scatter it.
        local_k = dk[:, :, rank*4:(rank+1)*4]
        local_v = dv[:, :, rank*4:(rank+1)*4]
        send = torch.stack([dk[:, :, (1-rank)*4:(2-rank)*4], dv[:, :, (1-rank)*4:(2-rank)*4]])
    receive = torch.empty(2, 6, peer_end-peer_start, 4, 128, **bf)
    ops = []
    if local_len:
        ops.append(dist.P2POp(dist.isend, send, peer, group))
    if peer_end > peer_start:
        ops.append(dist.P2POp(dist.irecv, receive, peer, group))
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    pool = worker.draft_model_runner.token_to_kv_pool
    local_loc = cache_loc[start:end]
    peer_loc = cache_loc[peer_start:peer_end]
    for layer in range(6):
        if local_len:
            pool.set_kv_buffer(None, local_loc, local_k[layer], local_v[layer],
                               layer_id_override=layer)
        if peer_end > peer_start:
            pool.set_kv_buffer(None, peer_loc, receive[0, layer], receive[1, layer],
                               layer_id_override=layer)
