from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_quant_permute_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    src2dst_ptr,
    HIDDEN_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    input_offsets = token_idx * HIDDEN_SIZE + offsets

    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(values)), 1.0e-30)
    scale = absmax / 448.0
    quantized = (values / scale).to(tl.float8e4nv)

    token_src2dst = src2dst_ptr + token_idx * TOPK
    for topk_idx in tl.static_range(TOPK):
        dst_idx = tl.load(token_src2dst + topk_idx).to(tl.int64)
        if dst_idx >= 0:
            output_offsets = dst_idx * HIDDEN_SIZE + offsets
            tl.store(output_ptr + output_offsets, quantized, mask=mask)
            tl.store(scale_ptr + dst_idx, scale)


def fused_quant_permute_fp8(
    inputs: torch.Tensor,
    src2dst: torch.Tensor,
    topk: int,
    outputs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each source token once and scatter FP8 rows by expert order."""

    if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"inputs must be floating point, got {inputs.dtype}")
    if not inputs.is_cuda or not src2dst.is_cuda:
        raise ValueError("inputs and src2dst must be CUDA tensors")
    if inputs.dim() != 2:
        raise ValueError(f"inputs must be 2D, got shape {tuple(inputs.shape)}")
    if src2dst.dtype != torch.int32:
        raise TypeError(f"src2dst must be int32, got {src2dst.dtype}")
    if not inputs.is_contiguous() or not src2dst.is_contiguous():
        raise ValueError("inputs and src2dst must be contiguous")
    if src2dst.numel() != inputs.size(0) * topk:
        raise ValueError("src2dst size must equal num_tokens * topk")

    output_shape = (src2dst.numel(), inputs.size(1))
    scale_shape = (src2dst.numel(), 1)
    if outputs is None:
        outputs = torch.empty(
            output_shape, dtype=torch.float8_e4m3fn, device=inputs.device
        )
    if scales is None:
        scales = torch.empty(scale_shape, dtype=torch.float32, device=inputs.device)

    if outputs.shape != output_shape or outputs.dtype != torch.float8_e4m3fn:
        raise ValueError("outputs must be contiguous FP8 E4M3 with expert-order shape")
    if scales.shape != scale_shape or scales.dtype != torch.float32:
        raise ValueError("scales must be contiguous FP32 with shape [routed_rows, 1]")
    if not outputs.is_contiguous() or not scales.is_contiguous():
        raise ValueError("outputs and scales must be contiguous")

    hidden_size = inputs.size(1)
    block_size = triton.next_power_of_2(hidden_size)
    effective_block = block_size
    num_warps = min(max(effective_block // 256, 1), 8)
    _fused_quant_permute_fp8_kernel[(inputs.size(0),)](
        inputs,
        outputs,
        scales,
        src2dst,
        HIDDEN_SIZE=hidden_size,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return outputs, scales
