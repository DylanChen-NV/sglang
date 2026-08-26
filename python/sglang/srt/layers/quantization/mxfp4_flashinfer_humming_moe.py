"""DeepSeek-V4 MXFP4 experts backed by FlashInfer PR #3738 Humming MoE."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

_PREPROCESS_EXPERT_CHUNK = int(
    os.environ.get("SGLANG_MXFP4_PREPROCESS_EXPERT_CHUNK", "32")
)
_TUNE_MAX_NUM_TOKENS = int(
    os.environ.get("SGLANG_W4AFP8_TUNE_MAX_NUM_TOKENS", "16384")
)
if _PREPROCESS_EXPERT_CHUNK < 1:
    raise ValueError("SGLANG_MXFP4_PREPROCESS_EXPERT_CHUNK must be positive")
if _TUNE_MAX_NUM_TOKENS < 1:
    raise ValueError("SGLANG_W4AFP8_TUNE_MAX_NUM_TOKENS must be positive")


class Mxfp4FlashinferHummingMoEMethod:
    """MXFP4 x dynamic-FP8 MoE using FlashInfer's SM90 Humming path.

    This is the FlashInfer PR #3738 implementation, not SGLang's generic
    ``humming`` package backend. It consumes the DSV4 checkpoint's packed E2M1
    weights and E8M0 group-32 scales, then rewrites them to the Humming layout.
    """

    _runtime_logged = False

    def __init__(self, fp8_method, prefix: str):
        if not is_sm90_supported():
            raise RuntimeError("flashinfer_humming requires an SM90 GPU")
        self._fp8 = fp8_method
        self.prefix = prefix
        self.moe_runner_config = None
        self._swiglu_limit_tensor: torch.Tensor | None = None

    @property
    def load_up_proj_weight_first(self) -> bool:
        """Load W13 as ``[up; gate]``, as required by fused_moe_90."""
        return True

    @staticmethod
    def _helpers():
        try:
            from flashinfer.fused_moe import (
                cutlass_fused_moe,
                preprocess_moe_weights_for_sm90_mixed_gemm_humming,
            )
            try:
                from flashinfer.fused_moe.core import ActivationType
            except ImportError:
                from flashinfer.tllm_enums import ActivationType
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "flashinfer_humming requires a FlashInfer build containing "
                "PR #3738 (validated with 0.6.16.post1)"
            ) from exc
        return (
            cutlass_fused_moe,
            preprocess_moe_weights_for_sm90_mixed_gemm_humming,
            ActivationType,
        )

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0:
            raise ValueError(
                "flashinfer_humming requires hidden and local intermediate "
                "dimensions divisible by 128; got "
                f"hidden={hidden_size}, intermediate={intermediate_size_per_partition}"
            )
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config
        if layer.moe_ep_size != 1:
            raise ValueError("flashinfer_humming currently requires EP1")
        if not moe_runner_config.is_gated:
            raise ValueError("flashinfer_humming currently requires gated SwiGLU MoE")

        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        if swiglu_limit is not None:
            self._swiglu_limit_tensor = torch.full(
                (layer.num_local_experts,),
                float(swiglu_limit),
                dtype=torch.float32,
                device=layer.w13_weight.device,
            )

    @staticmethod
    def _preprocess_in_chunks(preprocess, weight, raw_scale):
        outputs = ([], [], [])
        for start in range(0, weight.shape[0], _PREPROCESS_EXPERT_CHUNK):
            end = min(start + _PREPROCESS_EXPERT_CHUNK, weight.shape[0])
            chunks = preprocess(
                weight[start:end].contiguous(),
                raw_scale[start:end].contiguous(),
            )
            if len(chunks) != len(outputs):
                raise RuntimeError(
                    "FlashInfer Humming preprocess must return "
                    "(weight, folded_scale, residual)"
                )
            for parts, chunk in zip(outputs, chunks):
                parts.append(chunk)
        return tuple(torch.cat(parts, dim=0).contiguous() for parts in outputs)

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            raise RuntimeError("flashinfer_humming does not support MegaMoE weights")
        if layer.moe_ep_size != 1:
            raise ValueError("flashinfer_humming currently requires EP1")

        _, preprocess, _ = self._helpers()
        log_info_on_rank0(
            logger,
            f"Preparing DSV4 MXFP4 experts for FlashInfer Humming "
            f"(layer: {self.prefix})...",
        )
        for stem in ("w13", "w2"):
            raw_weight = getattr(layer, f"{stem}_weight")
            raw_scale_name = f"{stem}_weight_scale_inv"
            raw_scale = getattr(layer, raw_scale_name)
            if raw_weight.dtype != torch.int8 or not raw_weight.is_contiguous():
                raise TypeError(
                    f"{stem} raw MXFP4 weight must be contiguous int8, "
                    f"got {raw_weight.dtype}"
                )
            if raw_scale.dtype != torch.float8_e8m0fnu:
                raise TypeError(
                    f"{stem} raw MXFP4 scale must be E8M0, got {raw_scale.dtype}"
                )

            weight, folded_scale, residual = self._preprocess_in_chunks(
                preprocess,
                raw_weight.detach().view(torch.uint8),
                raw_scale.detach().view(torch.uint8),
            )
            setattr(layer, f"{stem}_weight", Parameter(weight, requires_grad=False))
            setattr(
                layer,
                f"{stem}_weight_scale",
                Parameter(folded_scale, requires_grad=False),
            )
            setattr(
                layer,
                f"{stem}_weight_residual",
                Parameter(residual, requires_grad=False),
            )
            delattr(layer, raw_scale_name)
            del raw_weight, raw_scale, weight, folded_scale, residual
            torch.cuda.empty_cache()

        layer.mxfp4_fc2_act_global = Parameter(
            torch.ones((), dtype=torch.float32, device=layer.w2_weight.device),
            requires_grad=False,
        )
        layer._dsv4_mxfp4_backend = "flashinfer_humming"

    @staticmethod
    def _expert_major_residual_scales(
        topk_ids: torch.Tensor,
        *expert_residuals: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        flat_ids = topk_ids.reshape(-1).to(torch.long)
        expert_major_order = torch.argsort(flat_ids, stable=True)
        result = []
        for residual in expert_residuals:
            route_scale = residual.index_select(0, flat_ids) * 64.0
            result.append(
                route_scale.index_select(0, expert_major_order).contiguous()
            )
        return tuple(result)

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        x = dispatch_output.hidden_states
        if x.dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "FlashInfer Humming dynamically quantizes BF16/FP16 inputs; "
                f"got {x.dtype}"
            )
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        routed_scaling_factor = self.moe_runner_config.routed_scaling_factor or 1.0
        if routed_scaling_factor != 1.0:
            topk_weights = topk_weights * routed_scaling_factor

        fc1_residual, fc2_residual = self._expert_major_residual_scales(
            topk_ids,
            layer.w13_weight_residual,
            layer.w2_weight_residual,
        )
        quant_scales = [
            layer.w13_weight_scale.view(torch.int32),
            fc1_residual,
            layer.mxfp4_fc2_act_global,
            layer.w2_weight_scale.view(torch.int32),
            fc2_residual,
        ]
        output = torch.empty_like(x)
        cutlass_fused_moe, _, ActivationType = self._helpers()

        if not Mxfp4FlashinferHummingMoEMethod._runtime_logged:
            log_info_on_rank0(
                logger,
                "Executing FlashInfer PR #3738 Humming MoE "
                f"with tune_max_num_tokens={_TUNE_MAX_NUM_TOKENS}",
            )
            Mxfp4FlashinferHummingMoEMethod._runtime_logged = True

        cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=x.dtype,
            quant_scales=quant_scales,
            output=output,
            swiglu_limit=self._swiglu_limit_tensor,
            use_w4_group_scaling=True,
            use_packed_weights=False,
            use_wfp4afp8_humming=True,
            activation_type=ActivationType.Swiglu,
            tp_size=layer.moe_tp_size,
            tp_rank=layer.moe_tp_rank,
            ep_size=1,
            ep_rank=0,
            profile_ids=None,
            tune_max_num_tokens=_TUNE_MAX_NUM_TOKENS,
        )
        return StandardCombineInput(hidden_states=output)
