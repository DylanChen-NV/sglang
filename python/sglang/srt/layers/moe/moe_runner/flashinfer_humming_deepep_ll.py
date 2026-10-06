"""Graph-safe DeepEP LL padded carrier adapter for FlashInfer Humming.

No LowLatencyGroupedGEMM dependency. Expert rows retain their physical order;
invalid slots are zeroed and routed to -1, so only valid rows enter GEMMs.
The fixed carrier size (including padding) is a correctness-first contract,
not a claim of optimal workspace/quantization cost for oversized capacity.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_ll_rows(
    X, SF, COUNTS, Y, YSF, IDS, WEIGHTS,
    CAPACITY: tl.constexpr, HIDDEN: tl.constexpr,
    X_E: tl.constexpr, X_M: tl.constexpr, X_H: tl.constexpr,
    SF_E: tl.constexpr, SF_M: tl.constexpr,
    FP8: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    expert = row // CAPACITY
    slot = row % CAPACITY
    valid = slot < tl.load(COUNTS + expert)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    if FP8:
        # Transport FP8 bytes unchanged. Besides avoiding an unnecessary
        # conversion, uint8 masking works on Triton versions that cannot
        # cast the integer `other=0` of tl.load directly into FP8 E4M3.
        src = X.to(tl.pointer_type(tl.uint8))
        dst = Y.to(tl.pointer_type(tl.uint8))
    else:
        src, dst = X, Y
    values = tl.load(
        src + expert * X_E + slot * X_M + cols * X_H,
        mask=valid & (cols < HIDDEN), other=0,
    )
    tl.store(dst + row * HIDDEN + cols, values, mask=cols < HIDDEN)
    if chunk == 0:
        tl.store(IDS + row, tl.where(valid, expert, -1))
        # DeepEP low_latency_combine applies the original router weights.
        tl.store(WEIGHTS + row, tl.where(valid, 1.0, 0.0))
        if FP8:
            scale = tl.load(
                SF + expert * SF_E + slot * SF_M, mask=valid, other=1.0,
            )
            tl.store(YSF + row, scale)


@triton.jit
def _zero_ll_padding(OUT, COUNTS, CAPACITY: tl.constexpr, HIDDEN: tl.constexpr,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0)
    expert = row // CAPACITY
    slot = row % CAPACITY
    if slot >= tl.load(COUNTS + expert):
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(OUT + row * HIDDEN + cols, 0.0, mask=cols < HIDDEN)


def prepare_ll_rows(dispatch_output, quant_info):
    """Return a standard top-k1 carrier without host reads of expert counts.

    FP8 scales must have been produced by the per-token DeepEP transport;
    the legacy group-128 carrier repeats that scale, so group zero is enough.
    Arbitrary group-128 quantization is NOT compatible with this interface.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    x, sf, counts = (
        dispatch_output.hidden_states,
        dispatch_output.hidden_states_scale,
        dispatch_output.masked_m,
    )
    if x.ndim != 3:
        raise ValueError("DeepEP LL Humming requires expert-major [E, capacity, H]")
    experts, capacity, hidden = x.shape
    if counts.shape != (experts,) or counts.dtype != torch.int32:
        raise ValueError("DeepEP LL Humming requires int32 expert counts shaped [E]")
    if counts.device != x.device:
        raise ValueError("DeepEP LL carrier and counts must be on the same device")
    if quant_info.w13_weight.shape[0] != experts:
        raise ValueError("DeepEP LL carrier and Humming local weight shard disagree")
    is_fp8 = x.dtype == torch.float8_e4m3fn
    if not ((x.dtype == torch.bfloat16 and sf is None) or is_fp8):
        raise ValueError("DeepEP LL Humming requires BF16 or per-token FP8 E4M3")
    if is_fp8 and (
        sf is None or sf.dtype != torch.float32 or sf.device != x.device
        or sf.ndim != 3 or tuple(sf.shape[:2]) != (experts, capacity)
        or sf.shape[2] not in (1, hidden // 128)
        or hidden % 128
    ):
        raise ValueError("DeepEP LL FP8 requires repeated per-token FP32 scales [E,C,H/128] or [E,C,1]")
    if not counts.is_contiguous():
        counts = counts.contiguous()
    rows = experts * capacity
    y = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
    ysf = torch.empty((rows, 1), dtype=torch.float32, device=x.device) if is_fp8 else None
    ids = torch.empty((rows, 1), dtype=torch.int32, device=x.device)
    weights = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    if rows:
        _prepare_ll_rows[(rows, triton.cdiv(hidden, 512))](
            x, sf if is_fp8 else x, counts, y, ysf if is_fp8 else y, ids, weights,
            capacity, hidden, *x.stride(),
            sf.stride(0) if is_fp8 else 0, sf.stride(1) if is_fp8 else 0,
            is_fp8, 512,
        )
    return StandardDispatchOutput(
        y, ysf, StandardTopKOutput(weights, ids, None)
    )


def run_deepep_ll_humming(dispatch_output, quant_info, runner_config, run_standard):
    from dataclasses import replace

    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLCombineInput

    humming_scales = (
        quant_info.w13_humming_residual_scale,
        quant_info.w2_humming_residual_scale,
        quant_info.humming_fc2_act_scale,
    )
    if not all(scale is not None for scale in humming_scales):
        raise ValueError("DeepEP LL requires FlashInfer SM90 Humming W4A8 weights")
    if runner_config.apply_router_weight_on_input:
        raise ValueError("DeepEP LL Humming applies router weights at combine, not input")
    standard = prepare_ll_rows(dispatch_output, quant_info)
    shape = dispatch_output.hidden_states.shape
    if standard.hidden_states.shape[0]:
        result = run_standard(
            standard, replace(quant_info, moe_ep_size=1, moe_ep_rank=0),
            # Router scaling belongs to DeepEP combine, not the local top-k1.
            replace(runner_config, routed_scaling_factor=None),
        ).hidden_states
        _zero_ll_padding[(shape[0] * shape[1], triton.cdiv(shape[2], 512))](
            result, dispatch_output.masked_m.contiguous(), shape[1], shape[2], 512,
        )
        result = result.view(shape)
    else:
        result = torch.empty(shape, dtype=torch.bfloat16, device=standard.hidden_states.device)
    return DeepEPLLCombineInput(result, dispatch_output.topk_ids, dispatch_output.topk_weights)
