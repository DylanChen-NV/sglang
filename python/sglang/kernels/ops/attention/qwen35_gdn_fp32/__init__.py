# SPDX-License-Identifier: Apache-2.0
"""Pinned SM100 GDN with an FP32 output side channel.

The CP rank-1 local output may preserve the pre-rounding accumulator. The
optional U producer instead stores FP32(BF16(acc)) in caller-owned scratch,
preserving the original response values. The stock BF16 epilogue remains for
bitwise regression; state/checkpoints follow the archived FlashInfer 0.7.1.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _check_runtime():
    import flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked as upstream
    expected = {
        "gated_delta_net_chunked.py": "f5db6379519b82b6e293d1f95560fa75af3a19a59c5aeb3aa00cc798df3957bc",
        "gdn_prefill.py": "02da221978b5a2e1435b7a9a5edbb96ee7abcdcf80bc77f403ae9fa9d09fa3ec",
        "gated_delta_net_tile_scheduler.py": "96654e5e496907907f211b0768a993144b1793c094d2dd9919a5bc78502ed4a9",
        "gated_delta_net_cp_prefill.py": "91bc85a33f9742a0d9f5b4eb256f6cb41e81f17f61172d43d23e85d61c6e0408",
        "gdn_cp_prefill.py": "b2cac9c06613672ec4a3e8bdfc44f1301b246ca078fc67a264d34ed49c986898",
    }
    root = Path(upstream.__file__).parent
    for name, digest in expected.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("FP32 local GDN requires the pinned FlashInfer source: " + name)


def chunk_gated_delta_rule_fp32_output(*, q, k, v, g, beta, scale=None,
        initial_state, output_final_state=True, cu_seqlens,
        use_qk_l2norm_in_kernel=False, output_state, state_checkpoints=None,
        checkpoint_cu_starts=None, checkpoint_every_n_tokens=0,
        return_bf16_output=False, output_fp32_buffer=None,
        round_output_to_bf16=False, emit_bf16_output=True):
    _check_runtime()
    if (torch.cuda.get_device_capability(q.device)[0] != 10
            or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
            or q.shape[-1] != 128 or tuple(v.shape[1:]) != (32, 128)
            or initial_state.dtype != torch.float32 or output_state.dtype != torch.float32
            or use_qk_l2norm_in_kernel or not output_final_state
            or not all(t.is_contiguous() for t in (q,k,v,g,beta,initial_state,output_state))):
        raise ValueError("FP32 local GDN requires contiguous SM100 BF16 inputs and FP32 state")
    if return_bf16_output and not emit_bf16_output:
        raise ValueError("Cannot return the disabled BF16 side output")
    # The descriptor still needs the original shape. With stores disabled,
    # V is a read-only placeholder; no BF16 output allocation is needed.
    out16 = torch.empty_like(v) if emit_bf16_output else v
    out32 = output_fp32_buffer
    if out32 is None:
        out32 = torch.empty(v.shape, dtype=torch.float32, device=v.device)
    elif (out32.shape != v.shape or out32.dtype != torch.float32
            or out32.device != v.device or not out32.is_contiguous()):
        raise ValueError("FP32 GDN output buffer must be separate contiguous FP32 storage with V shape/device")
    from flashinfer.utils import get_device_sm_count, get_device_name
    from flashinfer.gdn_kernels.delta_rule_dsl.varlen_helper import should_use_cp_host
    use_cp = should_use_cp_host((len(cu_seqlens)-1)*32,
        get_device_sm_count(q.device),get_device_name(q.device),
        device_capability=torch.cuda.get_device_capability(q.device))
    if use_cp:
        from .gdn_cp_prefill import cp_delta_rule_dsl_sm100
        cp_delta_rule_dsl_sm100(out16,output_state,q,k,v,g,beta,cu_seqlens,
            scale or 128**-0.5,output_fp32=out32,initial_state=initial_state,
            state_checkpoints=state_checkpoints,checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,max_seqlen=len(q),
            round_output_to_bf16=round_output_to_bf16,
            emit_bf16_output=emit_bf16_output)
    else:
        from .gdn_prefill import chunk_gated_delta_rule_sm100
        chunk_gated_delta_rule_sm100(q,k,v,g,beta,out16,out32,cu_seqlens,
            initial_state,output_state,scale or 128**-0.5,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            cu_checkpoints=checkpoint_cu_starts,output_checkpoints=state_checkpoints,
            round_output_to_bf16=round_output_to_bf16,
            emit_bf16_output=emit_bf16_output)
    if return_bf16_output:
        return out32,output_state,out16
    return out32,output_state
