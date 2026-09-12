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
    """MXFP4 experts served by LowLatencyGroupedGEMM on SM90."""

    _VALID_VARIANTS = {"preopt", "s0", "final"}
    # CUDA Graph records every MoE layer sequentially. Reuse one fixed scratch
    # arena across method instances so full-model capture does not retain one
    # multi-GB DeepEP carrier workspace per layer.
    _SHARED_DEEPEP_LL_WORKSPACE_CACHE = {}

    def __init__(
        self, fp8_method, prefix: str, serialized_mxfp4: bool = False
    ):
        if not is_sm90_supported():
            raise RuntimeError("lowlatency_mxfp4 requires an SM90 GPU.")
        self._fp8 = fp8_method
        self._serialized_mxfp4 = serialized_mxfp4
        self.prefix = prefix
        # This backend executes its fused MoE path directly from apply().
        # Newer FusedMoE layers still expect every quant method to expose the
        # optional runner attribute for overlap hooks.
        self.runner = None
        self.variant = os.getenv("SGLANG_LOWLATENCY_MXFP4_VARIANT", "final").lower()
        if self.variant not in self._VALID_VARIANTS:
            raise ValueError(
                "SGLANG_LOWLATENCY_MXFP4_VARIANT must be one of "
                f"{sorted(self._VALID_VARIANTS)}, got {self.variant!r}."
            )
        persistent_ctas_override = os.getenv("SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS")
        self._persistent_ctas_fixed = persistent_ctas_override is not None
        self.persistent_ctas = int(persistent_ctas_override or "312")
        self._deepep_ll_offsets_cache = {}
        self._deepep_ll_workspace_cache = self._SHARED_DEEPEP_LL_WORKSPACE_CACHE
        self.deepep_layout = os.getenv(
            "SGLANG_LOWLATENCY_DEEPEP_LAYOUT", "compact"
        ).lower()
        if self.deepep_layout not in {"strided", "compact"}:
            raise ValueError(
                "SGLANG_LOWLATENCY_DEEPEP_LAYOUT must be strided or compact, "
                f"got {self.deepep_layout!r}."
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
        if num_experts <= 0:
            raise ValueError(
                f"lowlatency_mxfp4 requires a positive expert count, got {num_experts}."
            )
        if (
            hidden_size <= 0
            or hidden_size % 64 != 0
            or intermediate_size_per_partition <= 0
            or intermediate_size_per_partition % 64 != 0
        ):
            raise ValueError(
                "lowlatency_mxfp4 requires positive hidden and local intermediate "
                "sizes that are multiples of 64; got "
                f"hidden={hidden_size}, intermediate={intermediate_size_per_partition}."
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
        if moe_runner_config.top_k <= 0:
            raise ValueError(
                f"lowlatency_mxfp4 requires a positive top_k, got {moe_runner_config.top_k}."
            )
        log_info_on_rank0(
            logger,
            "Using lowlatency_mxfp4 "
            f"variant={self.variant}, persistent_ctas={self.persistent_ctas}, "
            f"deepep_layout={self.deepep_layout}",
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
        if (
            raw_weight.dtype not in (torch.int8, torch.uint8)
            or not raw_weight.is_contiguous()
        ):
            raise TypeError(
                "LowLatency raw MXFP4 weights must be contiguous int8 or uint8."
            )
        if (
            raw_scale.dtype not in (torch.float8_e8m0fnu, torch.uint8)
            or not raw_scale.is_contiguous()
        ):
            raise TypeError(
                "LowLatency raw MXFP4 scales must be contiguous E8M0 or uint8."
            )
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
        # Mxfp4Config already loaded the official packed E2M1 plus uint8 E8M0
        # tensors in the exact logical layout required here. Its default
        # post-load path would upcast those weights to BF16, so bypass it.
        if not self._serialized_mxfp4:
            self._fp8.process_weights_after_loading(layer)
        if getattr(layer, "_mega_moe_weights_built", False):
            raise RuntimeError("lowlatency_mxfp4 does not support MegaMoE weights.")
        if layer.num_local_experts <= 0:
            raise ValueError("lowlatency_mxfp4 requires local experts.")
        # Routing, weight preprocessing, and workspaces are parameterized by the
        # local expert count. EP1/256 experts is the validated DSV4 path; Kimi K3
        # uses 896 / ep_size local experts and is validated separately.

        total_start = time.perf_counter()
        for stem in ("w13", "w2"):
            raw_weight = getattr(layer, f"{stem}_weight")
            scale_inv_name = f"{stem}_weight_scale_inv"
            scale_name = (
                scale_inv_name
                if hasattr(layer, scale_inv_name)
                else f"{stem}_weight_scale"
            )
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
    def _workspace(
        rows: int, n: int, k: int, num_local_experts: int, device: torch.device
    ):
        return {
            "q": torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device),
            "counts": torch.empty(
                (num_local_experts,), dtype=torch.int32, device=device
            ),
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
        counts_precomputed: bool = False,
    ) -> torch.Tensor:
        import low_latency_mxfp4 as llop

        weight = getattr(layer, f"{stem}_weight")
        weight_offsets = getattr(layer, f"{stem}_weight_exp_offsets")
        residual = getattr(layer, f"{stem}_expert_residual")
        n = (
            2 * layer.intermediate_size_per_partition
            if stem == "w13"
            else layer.hidden_size
        )
        k = (
            layer.hidden_size
            if stem == "w13"
            else layer.intermediate_size_per_partition
        )
        if schedule is None:
            api = (
                llop.grouped_gemm_out_precomputed_counts
                if counts_precomputed
                else llop.grouped_gemm_out
            )
            return api(
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
        fc1 = self._workspace(
            rows,
            2 * layer.intermediate_size_per_partition,
            layer.hidden_size,
            layer.num_local_experts,
            hidden_states.device,
        )
        fc2 = self._workspace(
            rows,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            layer.num_local_experts,
            hidden_states.device,
        )
        compact = torch.empty(
            (rows, layer.hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        compact, src2dst, offsets = moe_permute(
            hidden_states,
            topk_ids,
            layer.num_local_experts,
            is_ep=False,
            outputs=compact,
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
        from sglang.kernels.ops.moe.moe_permute_prepare import (
            moe_permute_prepare_with_schedule,
        )
        from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

        rows = topk_ids.numel()
        fc1 = self._workspace(
            rows,
            2 * layer.intermediate_size_per_partition,
            layer.hidden_size,
            layer.num_local_experts,
            hidden_states.device,
        )
        fc2 = self._workspace(
            rows,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            layer.num_local_experts,
            hidden_states.device,
        )
        offsets, src2dst, counts, tile_experts, tile_n, num_tiles = (
            moe_permute_prepare_with_schedule(topk_ids, layer.num_local_experts)
        )
        schedule = (counts, tile_experts, tile_n, num_tiles)
        q1, s1 = fused_quant_permute_fp8(
            hidden_states,
            src2dst,
            topk_ids.size(1),
            outputs=fc1["q"],
            scales=fc1["token_scales"],
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

    def _run_deepep_ll_strided(
        self, layer: Module, dispatch_output
    ) -> torch.Tensor:
        """Consume DeepEP low-latency expert-major masked BF16 input.

        DeepEP gives every local expert a fixed expected_m stride and exposes
        valid prefix lengths through masked_m. LowLatency GEMM accepts separate
        row offsets and counts, so the adapter uses strided offsets directly;
        it does not sort or physically compact routed tokens.
        """
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            situ_and_mul,
        )
        from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
            act_and_mul_triton,
        )
        from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

        hidden_states = dispatch_output.hidden_states
        if dispatch_output.hidden_states_scale is not None:
            raise ValueError(
                "lowlatency_mxfp4 DeepEP low-latency initially supports BF16 "
                "communication only"
            )
        if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim not in (
            2,
            3,
        ):
            raise ValueError(
                "lowlatency_mxfp4 DeepEP low-latency requires a 2D or 3D BF16 "
                "expert-major tensor"
            )

        num_experts = layer.num_local_experts
        expected_m = int(dispatch_output.expected_m)
        if hidden_states.ndim == 3:
            if (
                hidden_states.shape[0] != num_experts
                or hidden_states.shape[2] != layer.hidden_size
            ):
                raise ValueError(
                    "DeepEP low-latency hidden-state shape does not match "
                    f"local_experts={num_experts}, hidden={layer.hidden_size}: "
                    f"got {tuple(hidden_states.shape)}"
                )
            capacity = int(hidden_states.shape[1])
        else:
            if (
                hidden_states.shape[1] != layer.hidden_size
                or hidden_states.shape[0] % num_experts
            ):
                raise ValueError(
                    "Flattened DeepEP low-latency hidden-state shape does not match "
                    f"local_experts={num_experts}, hidden={layer.hidden_size}: "
                    f"got {tuple(hidden_states.shape)}"
                )
            capacity = hidden_states.shape[0] // num_experts
        if expected_m < 0 or expected_m > capacity:
            raise ValueError(
                f"DeepEP expected_m={expected_m} exceeds expert capacity={capacity}"
            )
        rows = num_experts * capacity
        flat_hidden = hidden_states.reshape(rows, layer.hidden_size)
        masked_m = dispatch_output.masked_m.to(torch.int32).contiguous()
        if masked_m.numel() != num_experts:
            raise ValueError(
                f"DeepEP masked_m must have {num_experts} entries, "
                f"got {masked_m.numel()}"
            )

        cache_key = (hidden_states.device, num_experts, capacity)
        offsets = self._deepep_ll_offsets_cache.get(cache_key)
        if offsets is None:
            offsets = torch.arange(
                num_experts + 1,
                dtype=torch.int32,
                device=hidden_states.device,
            ).mul_(capacity)
            self._deepep_ll_offsets_cache[cache_key] = offsets

        fc1 = self._workspace(
            rows,
            2 * layer.intermediate_size_per_partition,
            layer.hidden_size,
            num_experts,
            hidden_states.device,
        )
        fc2 = self._workspace(
            rows,
            layer.hidden_size,
            layer.intermediate_size_per_partition,
            num_experts,
            hidden_states.device,
        )
        fc1["counts"].copy_(masked_m)
        q1, s1 = fc1["q"], fc1["token_scales"]
        sgl_per_token_quant_fp8(flat_hidden, q1, s1)
        gate_up = self._gemm(
            layer,
            "w13",
            q1,
            s1,
            offsets,
            fc1,
            counts_precomputed=True,
        )

        activated = torch.empty(
            (rows, layer.intermediate_size_per_partition),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        if self.moe_runner_config.activation == "situ":
            beta = self.moe_runner_config.gemm1_alpha
            linear_beta = self.moe_runner_config.gemm1_clamp_limit
            if beta is None or linear_beta is None:
                raise ValueError("Kimi K3 SiTU requires beta and linear_beta")
            situ_and_mul(activated, gate_up, beta, linear_beta)
        else:
            act_and_mul_triton(
                gate_up,
                activated,
                {},
                activation=self.moe_runner_config.activation,
                swiglu_limit=self.moe_runner_config.swiglu_limit,
            )

        q2, s2 = fc2["q"], fc2["token_scales"]
        sgl_per_token_quant_fp8(activated, q2, s2)
        down = self._gemm(
            layer,
            "w2",
            q2,
            s2,
            offsets,
            fc2,
            schedule=(
                fc1["counts"],
                fc1["tile_experts"],
                fc1["tile_n"],
                fc1["num_tiles"],
            ),
        )
        return down.view(num_experts, capacity, layer.hidden_size)

    def _run_deepep_ll_compact(
        self, layer: Module, dispatch_output
    ) -> torch.Tensor:
        """Run the B1/F1 compact-internal, padded-output pipeline.

        B1 accepts the padded BF16 carrier and quantizes it after dispatch.
        The per-token-equivalent F1 accepts DeepEP's padded E4M3 carrier whose
        group-128 scale slots repeat one token scale. LowLatency compacts valid
        rows byte-for-byte and collapses the repeated scale without requantizing.
        """
        from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8

        try:
            import low_latency_mxfp4 as llop
        except ImportError as exc:
            raise RuntimeError(
                "compact DeepEP LowLatency requires the matching "
                "LowLatencyGroupedGEMM extension"
            ) from exc
        if not hasattr(llop, "deepep_moe_out"):
            raise RuntimeError(
                "LowLatencyGroupedGEMM extension lacks deepep_moe_out; "
                "checkout the B1-compatible commit"
            )

        hidden_states = dispatch_output.hidden_states
        hidden_states_scale = dispatch_output.hidden_states_scale
        use_deepep_fp8 = hidden_states_scale is not None
        if use_deepep_fp8:
            if not hasattr(llop, "deepep_per_token_fp8_moe_out"):
                raise RuntimeError(
                    "LowLatencyGroupedGEMM extension lacks "
                    "deepep_per_token_fp8_moe_out; checkout the "
                    "per-token-equivalent F1 commit"
                )
            if hidden_states.dtype != torch.float8_e4m3fn or hidden_states.ndim != 3:
                raise ValueError(
                    "per-token F1 requires a 3D E4M3 expert-major carrier"
                )
        elif hidden_states.dtype != torch.bfloat16 or hidden_states.ndim not in (2, 3):
            raise ValueError(
                "B1 DeepEP LowLatency requires a 2D or 3D BF16 expert-major carrier"
            )
        num_experts = layer.num_local_experts
        if hidden_states.ndim == 3:
            if (
                hidden_states.shape[0] != num_experts
                or hidden_states.shape[2] != layer.hidden_size
            ):
                raise ValueError(
                    "DeepEP low-latency carrier does not match local experts "
                    f"and hidden size: {tuple(hidden_states.shape)}"
                )
            capacity = int(hidden_states.shape[1])
        else:
            if (
                hidden_states.shape[1] != layer.hidden_size
                or hidden_states.shape[0] % num_experts
            ):
                raise ValueError(
                    "Flattened DeepEP carrier has an invalid expert-major shape: "
                    f"{tuple(hidden_states.shape)}"
                )
            capacity = hidden_states.shape[0] // num_experts
        expected_m = int(dispatch_output.expected_m)
        if expected_m < 0 or expected_m > capacity:
            raise ValueError(
                f"DeepEP expected_m={expected_m} exceeds capacity={capacity}"
            )
        rows = num_experts * capacity
        # DeepEP low-latency supports fewer than 256 decode tokens per rank.
        # FC1 writes only compact routed rows, so its intermediates need at
        # most 256 * top_k rows instead of the padded E * capacity carrier.
        compact_rows = min(rows, 256 * self.moe_runner_config.top_k)
        if expected_m * num_experts > compact_rows:
            raise ValueError(
                "DeepEP low-latency compact workspace supports fewer than "
                "256 decode tokens per rank"
            )
        if use_deepep_fp8:
            expected_scale_shape = (
                num_experts, capacity, layer.hidden_size // 128,
            )
            if (
                layer.hidden_size % 128
                or hidden_states_scale.dtype != torch.float32
                or tuple(hidden_states_scale.shape) != expected_scale_shape
            ):
                raise ValueError(
                    "per-token F1 requires repeated DeepEP FP32 scales shaped "
                    f"{expected_scale_shape}, got dtype={hidden_states_scale.dtype}, "
                    f"shape={tuple(hidden_states_scale.shape)}"
                )
        masked_m = dispatch_output.masked_m.to(torch.int32).contiguous()
        if masked_m.numel() != num_experts:
            raise ValueError(
                f"DeepEP masked_m must have {num_experts} entries, "
                f"got {masked_m.numel()}"
            )
        beta = self.moe_runner_config.gemm1_alpha
        linear_beta = self.moe_runner_config.gemm1_clamp_limit
        if (
            self.moe_runner_config.activation != "situ"
            or beta is None
            or linear_beta is None
        ):
            raise ValueError(
                "compact DeepEP LowLatency currently requires Kimi K3 SiTU "
                "with beta and linear_beta"
            )

        cache_key = (
            hidden_states.device, num_experts, capacity, layer.hidden_size,
            layer.intermediate_size_per_partition,
        )
        workspace = self._deepep_ll_workspace_cache.get(cache_key)
        if workspace is None:
            int_options = dict(dtype=torch.int32, device=hidden_states.device)
            float_options = dict(dtype=torch.float32, device=hidden_states.device)
            workspace = {
                "q1": torch.empty(
                    (rows, layer.hidden_size),
                    dtype=torch.float8_e4m3fn, device=hidden_states.device,
                ),
                "q1_scales": torch.empty((rows, 1), **float_options),
                "padded_offsets": torch.empty((num_experts + 1,), **int_options),
                "compact_offsets": torch.empty((num_experts + 1,), **int_options),
                "tile_experts": torch.empty((rows,), **int_options),
                "tile_n": torch.empty((rows,), **int_options),
                "num_tiles": torch.empty((1,), **int_options),
                "fc1_token_scales": torch.empty((rows,), **float_options),
                "gate_up": torch.empty(
                    (compact_rows, 2 * layer.intermediate_size_per_partition),
                    dtype=torch.bfloat16, device=hidden_states.device,
                ),
                "q2": torch.empty(
                    (compact_rows, layer.intermediate_size_per_partition),
                    dtype=torch.float8_e4m3fn, device=hidden_states.device,
                ),
                "q2_scales": torch.empty((compact_rows, 1), **float_options),
                "fc2_token_scales": torch.empty(
                    (compact_rows,), **float_options
                ),
                "out": torch.empty(
                    (rows, layer.hidden_size),
                    dtype=torch.bfloat16, device=hidden_states.device,
                ),
            }
            self._deepep_ll_workspace_cache[cache_key] = workspace

        common_args = (
            layer.w13_weight, layer.w13_weight_exp_offsets,
            layer.w13_expert_residual, layer.w2_weight,
            layer.w2_weight_exp_offsets, layer.w2_expert_residual,
            masked_m, workspace["padded_offsets"], workspace["compact_offsets"],
            workspace["tile_experts"], workspace["tile_n"],
            workspace["num_tiles"],
        )
        common_tail = (
            workspace["fc1_token_scales"], workspace["gate_up"],
            workspace["q2"], workspace["q2_scales"],
            workspace["fc2_token_scales"], workspace["out"], capacity,
            layer.hidden_size, layer.intermediate_size_per_partition,
            self.persistent_ctas, float(beta), float(linear_beta),
        )
        if use_deepep_fp8:
            llop.deepep_per_token_fp8_moe_out(
                hidden_states, hidden_states_scale, *common_args,
                workspace["q1"], workspace["q1_scales"], *common_tail,
            )
        else:
            flat_hidden = hidden_states.reshape(rows, layer.hidden_size)
            sgl_per_token_quant_fp8(
                flat_hidden, workspace["q1"], workspace["q1_scales"]
            )
            llop.deepep_moe_out(
                workspace["q1"], workspace["q1_scales"],
                *common_args, *common_tail,
            )
        return workspace["out"].view(
            num_experts, capacity, layer.hidden_size
        )

    def _run_deepep_ll(self, layer: Module, dispatch_output) -> torch.Tensor:
        if self.deepep_layout == "strided":
            return self._run_deepep_ll_strided(layer, dispatch_output)
        return self._run_deepep_ll_compact(layer, dispatch_output)

    def apply(self, layer: Module, dispatch_output: DispatchOutput) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        if DispatchOutputChecker.format_is_deepep_ll(dispatch_output):
            return DeepEPLLCombineInput(
                hidden_states=self._run_deepep_ll(layer, dispatch_output),
                topk_ids=dispatch_output.topk_ids,
                topk_weights=dispatch_output.topk_weights,
            )
        if not DispatchOutputChecker.format_is_standard(dispatch_output):
            raise ValueError(
                "Unsupported lowlatency_mxfp4 dispatch format: "
                f"{dispatch_output.format}"
            )

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        hidden_states = dispatch_output.hidden_states
        topk_ids = topk_output.topk_ids.contiguous().to(torch.int32)
        topk_weights = topk_output.topk_weights
        use_preopt = self.variant == "preopt" or topk_ids.numel() > 64

        def run():
            if use_preopt:
                return self._run_preopt(layer, hidden_states, topk_ids, topk_weights)
            return self._run_optimized(layer, hidden_states, topk_ids, topk_weights)

        if not self._persistent_ctas_fixed and topk_ids.numel() <= 64:
            from sglang.srt.layers.quantization.lowlatency_mxfp4_autotune import (
                select_persistent_ctas,
            )

            def run_candidate(candidate: int):
                previous = self.persistent_ctas
                self.persistent_ctas = candidate
                try:
                    return run()
                finally:
                    self.persistent_ctas = previous

            self.persistent_ctas = select_persistent_ctas(
                variant=self.variant,
                rows=topk_ids.numel(),
                hidden_size=layer.hidden_size,
                intermediate_size=layer.intermediate_size_per_partition,
                default=self.persistent_ctas,
                run_candidate=run_candidate,
            )
            layer._lowlatency_persistent_ctas = self.persistent_ctas

        output = run()
        return StandardCombineInput(hidden_states=output)
