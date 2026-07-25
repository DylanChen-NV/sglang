from __future__ import annotations

import atexit
import logging
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


class NVFP4SM120AttnBackend(FlashInferAttnBackend):
    """Experimental dense NVFP4 prefill with a FlashInfer fallback.

    The NVFP4 kernel currently supports only square causal attention over
    freshly supplied Q/K/V. Equal-length requests in a fresh batch are
    bucketed into one dense kernel call; prefix/chunk/verify/decode paths
    retain the normal FlashInfer behavior.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        init_new_workspace: bool = False,
    ):
        super().__init__(
            model_runner,
            init_new_workspace=init_new_workspace,
        )
        # This subclass delegates every unsupported prefill path and all decode
        # paths to FlashInfer, so validate both fallback accesses explicitly.
        # FlashInfer's constructor normally performs these checks only for its
        # exact class.
        self._check_kv_attention_access("prefill", self.prefill_kv_access)
        self._check_kv_attention_access("decode", self.decode_kv_access)

        architectures = model_runner.model_config.hf_config.architectures or []
        if not any(name.startswith("Qwen3_5") for name in architectures):
            raise ValueError(
                "nvfp4_sm120 is an experiment scoped to Qwen3.5, "
                f"got architectures={architectures}"
            )
        capability = torch.cuda.get_device_capability()
        if capability not in ((12, 0), (12, 1)):
            raise ValueError(
                "nvfp4_sm120 requires compute capability 12.0 or 12.1, "
                f"got {capability}"
            )

        from flashinfer.nvfp4_attention_sm120 import (
            nvfp4_attention_sm120_fwd,
            nvfp4_attention_sm120_quantize_qkv,
        )

        self._nvfp4_fwd = nvfp4_attention_sm120_fwd
        self._nvfp4_quantize_qkv = nvfp4_attention_sm120_quantize_qkv
        self._nvfp4_is_draft_worker = model_runner.is_draft_worker
        self._nvfp4_per_block_mean = _env_flag(
            "SGLANG_NVFP4_ATTN_PER_BLOCK_MEAN", False
        )
        self._nvfp4_max_seq_len = int(
            os.getenv("SGLANG_NVFP4_ATTN_MAX_SEQ_LEN", "8192")
        )
        self._nvfp4_log_interval = max(
            1, int(os.getenv("SGLANG_NVFP4_ATTN_LOG_INTERVAL", "10"))
        )
        self._nvfp4_shadow_compare_limit = max(
            0, int(os.getenv("SGLANG_NVFP4_ATTN_SHADOW_COMPARE_CALLS", "0"))
        )
        self._nvfp4_shadow_compare_calls = 0
        self._nvfp4_profile = _env_flag("SGLANG_NVFP4_ATTN_PROFILE", False)
        self._nvfp4_context_parallel_size = get_parallel().attn_cp_size
        self._nvfp4_calls = 0
        self._nvfp4_tokens = 0
        self._nvfp4_kernel_launches = 0
        self._nvfp4_request_seq_lens: Counter[int] = Counter()
        self._nvfp4_scheduler_batch_sizes: Counter[int] = Counter()
        self._nvfp4_kernel_batch_sizes: Counter[int] = Counter()
        self._nvfp4_fallback_calls = 0
        self._nvfp4_fallback_tokens = 0
        self._nvfp4_fallback_reasons: Counter[str] = Counter()
        self._nvfp4_profiled_tokens = 0
        self._nvfp4_adapter_ms = 0.0
        self._nvfp4_quantize_ms = 0.0
        self._nvfp4_forward_ms = 0.0
        self._nvfp4_reported_calls = -1
        atexit.register(self._log_nvfp4_stats, True)

        logger.info(
            "Initialized nvfp4_sm120 prefill experiment: "
            "native_gqa=True, head_dim=256, per_block_mean=%s, "
            "max_seq_len=%d, equal_length_bucketing=True, profile=%s, "
            "shadow_compare_calls=%d, draft_worker=%s",
            self._nvfp4_per_block_mean,
            self._nvfp4_max_seq_len,
            self._nvfp4_profile,
            self._nvfp4_shadow_compare_limit,
            self._nvfp4_is_draft_worker,
        )

    def _log_nvfp4_stats(self, force: bool = False) -> None:
        total_calls = self._nvfp4_calls + self._nvfp4_fallback_calls
        if total_calls == 0 or self._nvfp4_reported_calls == total_calls:
            return
        if not force and total_calls % self._nvfp4_log_interval != 0:
            return
        self._nvfp4_reported_calls = total_calls
        logger.info(
            "nvfp4_sm120 stats: nvfp4_calls=%d nvfp4_tokens=%d "
            "kernel_launches=%d request_seq_lens=%s "
            "scheduler_batch_sizes=%s kernel_batch_sizes=%s "
            "fallback_calls=%d fallback_tokens=%d fallback_reasons=%s "
            "profiled_tokens=%d adapter_ms=%.6f quantize_ms=%.6f "
            "forward_ms=%.6f",
            self._nvfp4_calls,
            self._nvfp4_tokens,
            self._nvfp4_kernel_launches,
            dict(sorted(self._nvfp4_request_seq_lens.items())),
            dict(sorted(self._nvfp4_scheduler_batch_sizes.items())),
            dict(sorted(self._nvfp4_kernel_batch_sizes.items())),
            self._nvfp4_fallback_calls,
            self._nvfp4_fallback_tokens,
            dict(sorted(self._nvfp4_fallback_reasons.items())),
            self._nvfp4_profiled_tokens,
            self._nvfp4_adapter_ms,
            self._nvfp4_quantize_ms,
            self._nvfp4_forward_ms,
        )

    def _unsupported_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        kwargs: dict,
        save_kv_cache: bool,
    ) -> Optional[str]:
        if self._nvfp4_is_draft_worker:
            return "draft_worker"
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return f"forward_mode_{forward_batch.forward_mode.name.lower()}"
        if forward_batch.spec_info is not None:
            return "speculative_metadata"
        if torch.cuda.is_current_stream_capturing():
            return "cuda_graph_capture"
        if self._nvfp4_context_parallel_size != 1:
            return "context_parallel"
        if getattr(forward_batch, "mha_return_lse", False):
            return "return_lse"
        if kwargs:
            return "extra_attention_kwargs"
        if layer.is_cross_attention:
            return "cross_attention"
        if layer.attn_type != AttentionType.DECODER:
            return "non_decoder_attention"
        if layer.sliding_window_size != -1:
            return "sliding_window"
        if layer.logit_cap != 0:
            return "logit_cap"
        if (
            layer.qk_head_dim != 256
            or layer.v_head_dim != 256
            or layer.tp_k_head_num <= 0
            or layer.tp_q_head_num % layer.tp_k_head_num != 0
        ):
            return "unsupported_head_shape"
        if k is None or v is None:
            return "missing_kv"
        if q.ndim != 2 or k.ndim != 3 or v.ndim != 3:
            return "unexpected_qkv_rank"
        if q.shape[1] != layer.tp_q_head_num * layer.qk_head_dim:
            return "unexpected_q_shape"
        if tuple(k.shape[1:]) != (layer.tp_k_head_num, layer.qk_head_dim):
            return "unexpected_k_shape"
        if tuple(v.shape[1:]) != (layer.tp_v_head_num, layer.v_head_dim):
            return "unexpected_v_shape"
        if k.shape[0] != q.shape[0] or v.shape[0] != q.shape[0]:
            return "qkv_token_count_mismatch"
        if q.dtype not in (torch.float16, torch.bfloat16):
            return "unsupported_q_dtype"
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return "mixed_qkv_dtype"
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        seq_lens = forward_batch.extend_seq_lens_cpu
        if prefix_lens is None or seq_lens is None:
            return "missing_extend_cpu_metadata"
        if len(prefix_lens) != len(seq_lens) or len(seq_lens) == 0:
            return "invalid_extend_cpu_metadata"
        if any(int(prefix_len) != 0 for prefix_len in prefix_lens):
            return "cached_prefix"
        seq_lens_int = [int(seq_len) for seq_len in seq_lens]
        if any(seq_len <= 0 for seq_len in seq_lens_int):
            return "empty_sequence"
        if max(seq_lens_int) > self._nvfp4_max_seq_len:
            return "sequence_too_long"
        if sum(seq_lens_int) != q.shape[0]:
            return "token_count_mismatch"
        if save_kv_cache and forward_batch.out_cache_loc is None:
            return "missing_cache_location"
        return None

    def _record_fallback(self, reason: str, num_tokens: int) -> None:
        self._nvfp4_fallback_calls += 1
        self._nvfp4_fallback_tokens += num_tokens
        self._nvfp4_fallback_reasons[reason] += 1
        self._log_nvfp4_stats()

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        reason = self._unsupported_reason(
            q,
            k,
            v,
            layer,
            forward_batch,
            kwargs,
            save_kv_cache=save_kv_cache,
        )
        if reason is not None:
            self._record_fallback(reason, q.shape[0])
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
            )

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(
                    forward_batch.out_cache_loc,
                    self.forward_metadata.swa_out_cache_loc,
                ),
                k,
                v,
                *self._kv_write_scales(layer),
            )

        reference = None
        if self._nvfp4_shadow_compare_calls < self._nvfp4_shadow_compare_limit and (
            save_kv_cache or self.forward_metadata.use_ragged
        ):
            # The paged FlashInfer path reads K/V from the cache even for a
            # fresh prefill. Populate the cache first, then suppress its
            # duplicate write in the shadow call.
            reference = super().forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=False,
            )

        num_q_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        q_view = q.reshape(-1, num_q_heads, head_dim)
        k_view = k.reshape(-1, num_kv_heads, head_dim)
        v_view = v.reshape(-1, num_kv_heads, head_dim)
        seq_lens = [int(value) for value in forward_batch.extend_seq_lens_cpu]
        segments = []
        start = 0
        for request_index, seq_len in enumerate(seq_lens):
            end = start + seq_len
            segments.append((request_index, start, end, seq_len))
            start = end

        buckets = defaultdict(list)
        for segment in segments:
            buckets[segment[3]].append(segment)

        outputs: list[Optional[torch.Tensor]] = [None] * len(segments)
        adapter_ms = 0.0
        quantize_ms = 0.0
        forward_ms = 0.0
        profile_events = []
        for seq_len, bucket in buckets.items():
            if self._nvfp4_profile:
                pre_adapter_start = torch.cuda.Event(enable_timing=True)
                pre_adapter_end = torch.cuda.Event(enable_timing=True)
                quantize_start = torch.cuda.Event(enable_timing=True)
                quantize_end = torch.cuda.Event(enable_timing=True)
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                post_adapter_start = torch.cuda.Event(enable_timing=True)
                post_adapter_end = torch.cuda.Event(enable_timing=True)
                pre_adapter_start.record()

            q_dense = torch.stack(
                [
                    q_view[start:end].transpose(0, 1)
                    for _, start, end, _ in bucket
                ],
                dim=0,
            ).contiguous()
            k_dense = torch.stack(
                [
                    k_view[start:end].transpose(0, 1)
                    for _, start, end, _ in bucket
                ],
                dim=0,
            ).contiguous()
            v_dense = torch.stack(
                [
                    v_view[start:end].transpose(0, 1)
                    for _, start, end, _ in bucket
                ],
                dim=0,
            ).contiguous()
            if self._nvfp4_profile:
                pre_adapter_end.record()
                quantize_start.record()

            quantized = self._nvfp4_quantize_qkv(
                q_dense,
                k_dense,
                v_dense,
                per_block_mean=self._nvfp4_per_block_mean,
            )
            if self._nvfp4_profile:
                quantize_end.record()
                forward_start.record()

            output, _ = self._nvfp4_fwd(
                *quantized,
                sm_scale=layer.scaling,
                causal=True,
                per_block_mean=self._nvfp4_per_block_mean,
                out_dtype=v.dtype,
            )
            if self._nvfp4_profile:
                forward_end.record()
                post_adapter_start.record()

            for batch_index, (request_index, _, _, _) in enumerate(bucket):
                outputs[request_index] = (
                    output[batch_index, :, :seq_len]
                    .transpose(0, 1)
                    .contiguous()
                    .view(seq_len, num_q_heads * head_dim)
                )

            if self._nvfp4_profile:
                post_adapter_end.record()
                profile_events.append(
                    (
                        pre_adapter_start,
                        pre_adapter_end,
                        quantize_start,
                        quantize_end,
                        forward_start,
                        forward_end,
                        post_adapter_start,
                        post_adapter_end,
                    )
                )

            self._nvfp4_kernel_launches += 1
            self._nvfp4_kernel_batch_sizes[len(bucket)] += 1

        if profile_events:
            # All events are on the current stream. Waiting only for the final
            # event keeps multiple equal-length buckets asynchronous with one
            # another while still making the aggregate measurements exact.
            profile_events[-1][-1].synchronize()
            for (
                pre_adapter_start,
                pre_adapter_end,
                quantize_start,
                quantize_end,
                forward_start,
                forward_end,
                post_adapter_start,
                post_adapter_end,
            ) in profile_events:
                adapter_ms += pre_adapter_start.elapsed_time(
                    pre_adapter_end
                ) + post_adapter_start.elapsed_time(post_adapter_end)
                quantize_ms += quantize_start.elapsed_time(quantize_end)
                forward_ms += forward_start.elapsed_time(forward_end)

        self._nvfp4_calls += 1
        self._nvfp4_tokens += q.shape[0]
        self._nvfp4_scheduler_batch_sizes[len(seq_lens)] += 1
        self._nvfp4_request_seq_lens.update(seq_lens)
        if self._nvfp4_profile:
            self._nvfp4_profiled_tokens += q.shape[0]
            self._nvfp4_adapter_ms += adapter_ms
            self._nvfp4_quantize_ms += quantize_ms
            self._nvfp4_forward_ms += forward_ms
        self._log_nvfp4_stats()
        if any(output is None for output in outputs):
            raise RuntimeError("nvfp4_sm120 failed to materialize every batch output")
        materialized_outputs = [
            output for output in outputs if output is not None
        ]
        result = (
            materialized_outputs[0]
            if len(materialized_outputs) == 1
            else torch.cat(materialized_outputs, dim=0)
        )

        if reference is not None:
            output_fp32 = result.float()
            reference_fp32 = reference.float()
            diff = output_fp32 - reference_fp32
            output_flat = output_fp32.reshape(-1)
            reference_flat = reference_fp32.reshape(-1)
            output_norm = torch.linalg.vector_norm(output_flat)
            reference_norm = torch.linalg.vector_norm(reference_flat)
            diff_norm = torch.linalg.vector_norm(diff)
            denominator = output_norm * reference_norm
            cosine = torch.dot(output_flat, reference_flat) / denominator
            self._nvfp4_shadow_compare_calls += 1
            logger.info(
                "nvfp4_sm120 shadow: call=%d layer=%d seq_lens=%s "
                "mae=%.6f relative_l2=%.6f cosine=%.8f "
                "output_rms=%.6f reference_rms=%.6f "
                "q_rms=%.6f k_rms=%.6f v_rms=%.6f "
                "output_finite=%s reference_finite=%s",
                self._nvfp4_shadow_compare_calls,
                layer.layer_id,
                [int(value) for value in forward_batch.extend_seq_lens_cpu],
                diff.abs().mean().item(),
                (diff_norm / reference_norm).item(),
                cosine.item(),
                output_fp32.square().mean().sqrt().item(),
                reference_fp32.square().mean().sqrt().item(),
                q.float().square().mean().sqrt().item(),
                k.float().square().mean().sqrt().item(),
                v.float().square().mean().sqrt().item(),
                torch.isfinite(output_fp32).all().item(),
                torch.isfinite(reference_fp32).all().item(),
            )

        return result
