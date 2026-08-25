from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.utils import log_info_on_rank0
from sglang.srt.utils.common import is_sm90_supported

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class Mxfp4LowLatencyMoEMethod:
    """DSV4 MXFP4 experts served by LowLatencyGroupedGEMM on SM90."""

    _VALID_VARIANTS = {"preopt", "s0", "final"}

    def __init__(self, fp8_method, prefix: str):
        if not is_sm90_supported():
            raise RuntimeError("lowlatency_mxfp4 requires an SM90 GPU.")
        self._fp8 = fp8_method
        self.prefix = prefix
        self.variant = os.getenv("SGLANG_LOWLATENCY_MXFP4_VARIANT", "final").lower()
        if self.variant not in self._VALID_VARIANTS:
            raise ValueError(
                "SGLANG_LOWLATENCY_MXFP4_VARIANT must be one of "
                f"{sorted(self._VALID_VARIANTS)}, got {self.variant!r}."
            )
        self.persistent_ctas = int(
            os.getenv("SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS", "312")
        )
        if self.persistent_ctas <= 0:
            raise ValueError("LowLatency persistent_ctas must be positive.")

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if num_experts != 256:
            raise ValueError(f"lowlatency_mxfp4 requires 256 experts, got {num_experts}.")
        if hidden_size != 4096 or intermediate_size_per_partition != 512:
            raise ValueError(
                "lowlatency_mxfp4 currently supports DSV4 TP4 shapes only: "
                f"hidden=4096, intermediate=512; got {hidden_size}, "
                f"{intermediate_size_per_partition}."
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
        if moe_runner_config.top_k != 6:
            raise ValueError(
                f"lowlatency_mxfp4 requires top_k=6, got {moe_runner_config.top_k}."
            )
        log_info_on_rank0(
            logger,
            "Using lowlatency_mxfp4 "
            f"variant={self.variant}, persistent_ctas={self.persistent_ctas}",
        )

    @staticmethod
    def _preprocess_one(raw_weight: torch.Tensor, raw_scale: torch.Tensor):
        try:
            import low_latency_mxfp4 as llop
        except ImportError as exc:
            raise RuntimeError(
                "lowlatency_mxfp4 backend requires the LowLatencyGroupedGEMM "
                "extension on PYTHONPATH."
            ) from exc
        if raw_weight.dtype != torch.int8 or not raw_weight.is_contiguous():
            raise TypeError("LowLatency raw MXFP4 weights must be contiguous int8.")
        if raw_scale.dtype != torch.float8_e8m0fnu or not raw_scale.is_contiguous():
            raise TypeError("LowLatency raw MXFP4 scales must be contiguous E8M0.")
        experts, n, packed_k = raw_weight.shape
        if raw_scale.shape != (experts, n, packed_k // 16):
            raise ValueError(
                f"MXFP4 weight/scale shape mismatch: {tuple(raw_weight.shape)} vs "
                f"{tuple(raw_scale.shape)}."
            )
        interleaved, exp_offsets, residual = llop.preprocess_weight(
            raw_weight.view(torch.uint8), raw_scale.view(torch.uint8)
        )
        return (
            interleaved.view(experts, -1).contiguous(),
            exp_offsets.view(experts, -1).contiguous(),
            residual.contiguous(),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            raise RuntimeError("lowlatency_mxfp4 does not support MegaMoE weights.")
        if layer.num_local_experts != 256 or layer.moe_ep_size != 1:
            raise ValueError(
                "lowlatency_mxfp4 currently requires EP1 with all 256 experts local."
            )

        total_start = time.perf_counter()
        for stem in ("w13", "w2"):
            raw_weight = getattr(layer, f"{stem}_weight")
            scale_name = f"{stem}_weight_scale_inv"
            raw_scale = getattr(layer, scale_name)
            torch.cuda.synchronize(raw_weight.device)
            start = time.perf_counter()
            weight, offsets, residual = self._preprocess_one(raw_weight, raw_scale)
            torch.cuda.synchronize(raw_weight.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            setattr(layer, f"{stem}_weight", Parameter(weight, requires_grad=False))
            setattr(
                layer,
                f"{stem}_weight_exp_offsets",
                Parameter(offsets, requires_grad=False),
            )
            setattr(
                layer,
                f"{stem}_expert_residual",
                Parameter(residual, requires_grad=False),
            )
            delattr(layer, scale_name)
            del raw_weight, raw_scale, weight, offsets, residual
            torch.cuda.empty_cache()
            log_info_on_rank0(
                logger,
                f"LowLatency preprocess layer={self.prefix} tensor={stem} "
                f"elapsed_ms={elapsed_ms:.3f}",
            )

        layer._dsv4_mxfp4_backend = "lowlatency_mxfp4"
        layer._lowlatency_variant = self.variant
        layer._lowlatency_persistent_ctas = self.persistent_ctas
        log_info_on_rank0(
            logger,
            f"Prepared LowLatency MXFP4 layer={self.prefix} variant={self.variant} "
            f"total_ms={(time.perf_counter() - total_start) * 1000.0:.3f}",
        )

    @staticmethod
    def _workspace(rows: int, n: int, k: int, device: torch.device):
        return {
            "q": torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device),
            "counts": torch.empty((256,), dtype=torch.int32, device=device),
            "token_scales": torch.empty((rows, 1), dtype=torch.float32, device=device),
            "tile_experts": torch.empty((rows,), dtype=torch.int32, device=device),
            "tile_n": torch.empty((rows,), dtype=torch.int32, device=device),
            "num_tiles": torch.empty((1,), dtype=torch.int32, device=device),
            "out": torch.empty((rows, n), dtype=torch.bfloat16, device=device),
        }

    def _gemm(
        self,
        layer: Module,
        stem: str,
        q: torch.Tensor,
        scale: torch.Tensor,
        offsets: torch.Tensor,
        workspace: dict,
        schedule=None,
        scales_precombined: bool = False,
    ) -> torch.Tensor:
        import low_latency_mxfp4 as llop

        weight = getattr(layer, f"{stem}_weight")
        weight_offsets = getattr(layer, f"{stem}_weight_exp_offsets")
        residual = getattr(layer, f"{stem}_expert_residual")
        n = 2 * layer.intermediate_size_per_partition if stem == "w13" else layer.hidden_size
        k = layer.hidden_size if stem == "w13" else layer.intermediate_size_per_partition
        if schedule is None:
            return llop.grouped_gemm_out(
                q,
                scale,
                weight,
                weight_offsets,
                residual,
                offsets,
                workspace["counts"],
                workspace["token_scales"],
                workspace["tile_experts"],
                workspace["tile_n"],
                workspace["num_tiles"],
                workspace["out"],
                n,
                k,
                self.persistent_ctas,
            )
        api = (
            llop.grouped_gemm_out_precomputed_schedule_and_scales
            if scales_precombined
            else llop.grouped_gemm_out_precomputed_schedule
        )
        return api(
            q,
            scale,
            weight,
            weight_offsets,
            residual,
            offsets,
            schedule[0],
            workspace["token_scales"],
            schedule[1],
            schedule[2],
            schedule[3],
            workspace["out"],
            n,
            k,
            self.persistent_ctas,
        )

    def _run_preopt(self, layer: Module, hidden_states, topk_ids, topk_weights):
        from sglang.kernels.ops.moe.ep_moe_kernels import moe_permute, moe_unpermute
        from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import act_and_mul_triton

        rows = topk_ids.numel()
        fc1 = self._workspace(rows, 2 * layer.intermediate_size_per_partition, layer.hidden_size, hidden_states.device)
        fc2 = self._workspace(rows, layer.hidden_size, layer.intermediate_size_per_partition, hidden_states.device)
        compact = torch.empty((rows, layer.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        compact, src2dst, offsets = moe_permute(
            hidden_states, topk_ids, 256, is_ep=False, outputs=compact
        )
        q1, s1 = fc1["q"], fc1["token_scales"]
        sgl_per_token_quant_fp8(compact, q1, s1)
        gate_up = self._gemm(layer, "w13", q1, s1, offsets, fc1)
        activated = torch.empty(
            (rows, layer.intermediate_size_per_partition),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        act_and_mul_triton(
            gate_up,
            activated,
            {},
            activation=self.moe_runner_config.activation,
            swiglu_limit=self.moe_runner_config.swiglu_limit,
        )
        q2, s2 = fc2["q"], fc2["token_scales"]
        sgl_per_token_quant_fp8(activated, q2, s2)
        down = self._gemm(layer, "w2", q2, s2, offsets, fc2)
        return moe_unpermute(
            down,
            src2dst,
            topk_ids,
            topk_weights,
            self.moe_runner_config.routed_scaling_factor,
        )

    def _run_optimized(self, layer: Module, hidden_states, topk_ids, topk_weights):
        from sglang.kernels.ops.moe.ep_moe_kernels import moe_unpermute
        from sglang.kernels.ops.moe.fused_activation_quant import fused_swiglu_quant_fp8
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import act_and_mul_triton
        from sglang.kernels.ops.moe.fused_quant_permute import fused_quant_permute_fp8
        from sglang.kernels.ops.moe.moe_permute_prepare import moe_permute_prepare_with_schedule
        from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

        rows = topk_ids.numel()
        fc1 = self._workspace(rows, 2 * layer.intermediate_size_per_partition, layer.hidden_size, hidden_states.device)
        fc2 = self._workspace(rows, layer.hidden_size, layer.intermediate_size_per_partition, hidden_states.device)
        offsets, src2dst, counts, tile_experts, tile_n, num_tiles = (
            moe_permute_prepare_with_schedule(topk_ids, 256)
        )
        schedule = (counts, tile_experts, tile_n, num_tiles)
        q1, s1 = fused_quant_permute_fp8(
            hidden_states, src2dst, topk_ids.size(1), outputs=fc1["q"], scales=fc1["token_scales"]
        )
        gate_up = self._gemm(layer, "w13", q1, s1, offsets, fc1, schedule)
        if self.variant == "final":
            q2, s2 = fused_swiglu_quant_fp8(
                gate_up,
                offsets,
                layer.w2_expert_residual,
                self.moe_runner_config.swiglu_limit,
                outputs=fc2["q"],
                scales=fc2["token_scales"],
            )
            down = self._gemm(
                layer,
                "w2",
                q2,
                s2,
                offsets,
                fc2,
                schedule,
                scales_precombined=True,
            )
        else:
            activated = torch.empty(
                (rows, layer.intermediate_size_per_partition),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            act_and_mul_triton(
                gate_up,
                activated,
                {},
                activation=self.moe_runner_config.activation,
                swiglu_limit=self.moe_runner_config.swiglu_limit,
            )
            q2, s2 = fc2["q"], fc2["token_scales"]
            sgl_per_token_quant_fp8(activated, q2, s2)
            down = self._gemm(layer, "w2", q2, s2, offsets, fc2, schedule)
        return moe_unpermute(
            down,
            src2dst,
            topk_ids,
            topk_weights,
            self.moe_runner_config.routed_scaling_factor,
        )

    def apply(self, layer: Module, dispatch_output: DispatchOutput) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        hidden_states = dispatch_output.hidden_states
        topk_ids = topk_output.topk_ids.contiguous().to(torch.int32)
        topk_weights = topk_output.topk_weights
        if self.variant == "preopt" or topk_ids.numel() > 64:
            output = self._run_preopt(layer, hidden_states, topk_ids, topk_weights)
        else:
            output = self._run_optimized(layer, hidden_states, topk_ids, topk_weights)
        return StandardCombineInput(hidden_states=output)
