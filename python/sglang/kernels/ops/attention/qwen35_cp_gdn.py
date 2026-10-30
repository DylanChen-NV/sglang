# SPDX-License-Identifier: Apache-2.0
"""Reuse stock FlashInfer CP precompute within one local/homogeneous GDN pair.

The caller supplies already normalized, contiguous Q/K and a known-zero V
buffer. T(K,beta) and M(K,beta,alpha) are independent of V and initial state.
The homogeneous local N is zero. Buffers are native temporary workspaces, used
on the caller's stream and retained only for this one layer invocation.
Persistent recurrent/checkpoint state and the native kernels remain unchanged.
"""
import functools

import torch


@functools.lru_cache(maxsize=4)
def _metadata(device):
    from flashinfer.gdn_kernels.blackwell import gdn_cp_prefill as native
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
    from flashinfer.utils import get_device_name, get_device_sm_count
    from .qwen35_gdn_fp32 import _check_runtime

    _check_runtime()
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise ValueError("Qwen35 CP GDN pair is validated on SM100 only")
    sms, name = get_device_sm_count(device), get_device_name(device)
    use_cp = should_use_cp_host(32, sms, name, device_capability=capability)
    return native, capability, sms, name, use_cp


def create_pair(q, k, alpha, beta, cu_seqlens, zero_v):
    """Return a private two-call executor, or None for native non-CP dispatch."""
    native, capability, sms, name, use_cp = _metadata(q.device)
    if not use_cp:
        return None
    return _Pair(native, capability, sms, name, q, k, alpha, beta, cu_seqlens, zero_v)


class _Pair:
    def __init__(self, native, capability, sms, name, q, k, alpha, beta, cu, zero_v):
        import cuda.bindings.driver as cuda_driver

        rows = len(q)
        if (tuple(q.shape) != (rows, 16, 128) or k.shape != q.shape
                or tuple(zero_v.shape) != (rows, 32, 128)
                or tuple(alpha.shape) != (rows, 32) or beta.shape != alpha.shape
                or len(cu) != 2 or cu.dtype != torch.int32
                or any(t.dtype != torch.bfloat16 for t in (q, k, zero_v))
                or any(t.dtype != torch.float32 for t in (alpha, beta))
                or not all(t.is_contiguous() and t.device == q.device
                           for t in (q, k, alpha, beta, cu, zero_v))):
            raise ValueError("CP GDN pair requires the packed single-sequence Qwen35 BF16 layout")
        self.native, self.q, self.k = native, q, k
        self.alpha, self.beta, self.cu, self.zero_v = alpha, beta, cu, zero_v
        self.rows, self.device = rows, q.device
        self.stream = cuda_driver.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        self.chunk = native.choose_cp_chunk_len_host(
            rows, 32, sms, chunk_len_granularity=native.CP_CHUNK_LEN_GRANULARITY,
            device_capability=capability, total_seqlen=rows, num_seqs=1, device_name=name)
        self.calls = 0
        self.t = self.m = self.n = None

    def run(self, v, initial, *, output_state, output=None, homogeneous=False,
            state_checkpoints=None, checkpoint_cu_starts=None, checkpoint_every_n_tokens=0):
        if self.calls not in (0, 1) or homogeneous != (self.calls == 1):
            raise ValueError("CP GDN pair must execute local then homogeneous exactly once")
        if (v.shape != self.zero_v.shape or v.dtype != torch.bfloat16
                or v.device != self.device or not v.is_contiguous()
                or initial.dtype != torch.float32 or tuple(initial.shape) != (1, 32, 128, 128)
                or output_state.shape != initial.shape or output_state.dtype != initial.dtype
                or initial.device != self.device or output_state.device != self.device
                or not initial.is_contiguous() or not output_state.is_contiguous()):
            raise ValueError("CP GDN pair requires BF16 V and packed FP32 recurrent state")
        if homogeneous and v.data_ptr() != self.zero_v.data_ptr():
            raise ValueError("Homogeneous GDN must use the executor's zero V buffer")
        if output is None:
            output = torch.empty_like(v)
        elif (output.shape != v.shape or output.dtype != v.dtype or output.device != v.device
              or not output.is_contiguous() or output.data_ptr() == v.data_ptr()):
            raise ValueError("CP GDN output must be separate packed BF16 storage")
        native = self.native
        common = dict(_skip_check=True, _device=self.device, _stream=self.stream)
        with torch.cuda.nvtx.range("cp_delta_rule_sm100"):
            if not homogeneous:
                self.t = native.cp_delta_rule_t_precompute_dsl_sm100(
                    self.k, self.beta, self.cu, self.rows, max_seqlen=self.rows, **common)
                self.m, self.n = native.cp_delta_rule_mn_precompute_dsl_sm100(
                    self.k, v, self.t, self.alpha, self.cu, self.rows,
                    cp_chunk_len=self.chunk, max_seqlen=self.rows, **common)
            else:
                # Local GDN and its checkpoints have consumed N on this stream.
                # M/T are read-only in native fixup/prefill; V=0 gives local N=0.
                self.n.zero_()
            initial_workspace = native._get_cp_workspace(
                "gdn_cp_sm100_initial_state", (1, 32, 128, 128), torch.float32, self.device)
            fixed = native.cp_delta_rule_fixup_dsl_sm100(
                self.m, self.n, self.cu, self.rows, cp_chunk_len=self.chunk,
                initial_state=initial, output_state=output_state,
                initial_state_workspace=initial_workspace, **common)
            native.cp_delta_rule_prefill_dsl_sm100(
                output, None, self.q, self.k, v, self.t, fixed, self.alpha,
                128 ** -0.5, self.cu, self.rows, cp_chunk_len=self.chunk,
                max_seqlen=self.rows, initial_state=initial_workspace,
                state_checkpoints=state_checkpoints, checkpoint_cu_starts=checkpoint_cu_starts,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens, **common)
        self.calls += 1
        return output, output_state
