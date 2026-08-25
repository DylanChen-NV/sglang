from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_swiglu_quant_fp8_kernel(
    gate_up_ptr,
    output_ptr,
    scale_ptr,
    expert_offsets_ptr,
    expert_residual_ptr,
    INTERMEDIATE_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < INTERMEDIATE_SIZE
    row_base = row * (2 * INTERMEDIATE_SIZE)

    gate = tl.load(gate_up_ptr + row_base + cols, mask=mask, other=0.0)
    up = tl.load(
        gate_up_ptr + row_base + INTERMEDIATE_SIZE + cols,
        mask=mask,
        other=0.0,
    )
    gate = tl.minimum(gate, SWIGLU_LIMIT)
    up = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)

    # Preserve the original act_and_mul BF16 boundaries before quantization.
    gate_f32 = gate.to(tl.float32)
    gate_activated = (gate_f32 * tl.sigmoid(gate_f32)).to(tl.bfloat16)
    activated = (gate_activated * up).to(tl.bfloat16)
    activated_f32 = activated.to(tl.float32)

    absmax = tl.maximum(tl.max(tl.abs(activated_f32)), 1.0e-30)
    dequant_scale = absmax / 448.0
    quantized = (activated_f32 / dequant_scale).to(tl.float8e4nv)

    # Rows are expert-major. Locate the owning expert with a fixed-depth
    # lower_bound over offsets[1:NUM_EXPERTS+1].
    lo = tl.full((), 0, tl.int32)
    hi = tl.full((), NUM_EXPERTS, tl.int32)
    for _ in tl.static_range(8):
        mid = (lo + hi) // 2
        end = tl.load(expert_offsets_ptr + mid + 1)
        belongs_after_mid = row >= end
        lo = tl.where(belongs_after_mid, mid + 1, lo)
        hi = tl.where(belongs_after_mid, hi, mid)
    expert = lo
    combined_scale = dequant_scale * tl.load(expert_residual_ptr + expert) * 64.0

    tl.store(output_ptr + row * INTERMEDIATE_SIZE + cols, quantized, mask=mask)
    tl.store(scale_ptr + row, combined_scale)


def fused_swiglu_quant_fp8(
    gate_up: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_residual: torch.Tensor,
    swiglu_limit: float,
    outputs: torch.Tensor | None = None,
    scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse BF16 SwiGLU, per-row FP8 quant, and FC2 scale composition."""

    if gate_up.dtype != torch.bfloat16 or gate_up.dim() != 2:
        raise TypeError("gate_up must be a 2D BF16 tensor")
    if not gate_up.is_cuda or not gate_up.is_contiguous():
        raise ValueError("gate_up must be a contiguous CUDA tensor")
    if gate_up.size(1) % 2 != 0:
        raise ValueError("gate_up hidden dimension must be even")
    if expert_offsets.dtype != torch.int32 or expert_offsets.dim() != 1:
        raise TypeError("expert_offsets must be a 1D int32 tensor")
    if expert_residual.dtype != torch.float32 or expert_residual.dim() != 1:
        raise TypeError("expert_residual must be a 1D float32 tensor")
    if not expert_offsets.is_cuda or not expert_residual.is_cuda:
        raise ValueError("expert metadata must be CUDA tensors")
    if not expert_offsets.is_contiguous() or not expert_residual.is_contiguous():
        raise ValueError("expert metadata must be contiguous")
    num_experts = expert_residual.numel()
    if num_experts != 256 or expert_offsets.numel() != num_experts + 1:
        raise ValueError("A0 currently requires exactly 256 experts")

    rows = gate_up.size(0)
    intermediate_size = gate_up.size(1) // 2
    output_shape = (rows, intermediate_size)
    if outputs is None:
        outputs = torch.empty(
            output_shape, dtype=torch.float8_e4m3fn, device=gate_up.device
        )
    if scales is None:
        scales = torch.empty((rows, 1), dtype=torch.float32, device=gate_up.device)
    if outputs.shape != output_shape or outputs.dtype != torch.float8_e4m3fn:
        raise ValueError("outputs must be contiguous FP8 E4M3 with activated shape")
    if scales.shape != (rows, 1) or scales.dtype != torch.float32:
        raise ValueError("scales must be contiguous FP32 with shape [rows, 1]")
    if not outputs.is_contiguous() or not scales.is_contiguous():
        raise ValueError("outputs and scales must be contiguous")

    block_size = triton.next_power_of_2(intermediate_size)
    _fused_swiglu_quant_fp8_kernel[(rows,)](
        gate_up,
        outputs,
        scales,
        expert_offsets,
        expert_residual,
        INTERMEDIATE_SIZE=intermediate_size,
        NUM_EXPERTS=num_experts,
        SWIGLU_LIMIT=float(swiglu_limit),
        BLOCK_SIZE=block_size,
        num_warps=min(max(block_size // 256, 1), 8),
        num_stages=1,
    )
    return outputs, scales
