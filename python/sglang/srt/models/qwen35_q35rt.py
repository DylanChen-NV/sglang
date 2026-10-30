# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""q35rt compute/cache adaptation; scheduling and sampling stay in SGLang.

The compute backend is opt-in. Native SGLang owns request/cache lifetime,
sampling and acceptance; the leaf package owns its packed math implementation.
"""

from __future__ import annotations

import hashlib
import logging
import os

import torch
from torch import nn

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    KVCache,
    MHATokenToKVPool,
)

logger = logging.getLogger(__name__)


def enabled():
    return os.environ.get("SGLANG_Q35RT", "0") == "1"


def recurrent_state_layout():
    value = os.environ.get("SGLANG_Q35RT_SSM_LAYOUT", "vk")
    if value not in ("vk", "chunked_kv"):
        raise ValueError("SGLANG_Q35RT_SSM_LAYOUT must be vk or chunked_kv")
    return value


_ENGINES = {}


class Q35rtAttentionBackend(AttentionBackend):
    """q35rt reads native pool maps directly, without MHA/GDN metadata.

    Graph input loading, allocation, deferred Mamba clear/COW and checkpoint
    ownership stay on the normal ModelRunner/Scheduler paths. Only metadata
    consumed by native attention layers is absent: q35rt invokes none of them.
    """

    needs_cpu_seq_lens = False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        if max_bs != 1 or max_num_tokens != 8:
            raise ValueError("q35rt attention adapter requires the BS1/T8 graph")

    def get_cuda_graph_seq_len_fill_value(self):
        return 1


def build_q35rt_attention_backend():
    # Keep the prior metadata path available for controlled A/B measurements.
    value = os.environ.get("SGLANG_Q35RT_NATIVE_ATTN_METADATA", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_NATIVE_ATTN_METADATA must be 0 or 1")
    return None if value == "1" else Q35rtAttentionBackend()


class Q35rtNativeKVPool(MHATokenToKVPool):
    """Native transport layout plus the compute layout charged to its budget.

    Slabs are authoritative during decoding. Canonical buffers are for native
    transfer/offload only; receiving a prefix requires an explicit import.
    """

    def __init__(self, slab_key, slab_value, **kwargs):
        super().__init__(**kwargs)
        self.slab_key, self.slab_value = slab_key, slab_value
        if self.use_hnd:
            raise ValueError("q35rt native transfers require the NHD KV layout")
        self._native_key_ptrs = torch.tensor(
            [t.data_ptr() for t in self.k_buffer], dtype=torch.int64, device=self.device
        )
        self._native_value_ptrs = torch.tensor(
            [t.data_ptr() for t in self.v_buffer], dtype=torch.int64, device=self.device
        )
        self.mem_usage = sum(self.get_kv_size_bytes()) / (1 << 30)

    def import_locations(self, locations):
        from q35rt.cache_ops import exchange_native

        exchange_native(self.slab_key, locations, self._native_key_ptrs, True)
        exchange_native(self.slab_value, locations, self._native_value_ptrs, True)

    def export_locations(self, locations):
        from q35rt.cache_ops import exchange_native

        exchange_native(self.slab_key, locations, self._native_key_ptrs, False)
        exchange_native(self.slab_value, locations, self._native_value_ptrs, False)

    def move_kv_cache(self, tgt_loc, src_loc):
        from q35rt.cache_ops import gather, scatter

        scatter(self.slab_key, tgt_loc, gather(self.slab_key, src_loc))
        scatter(self.slab_value, tgt_loc, gather(self.slab_value, src_loc))

    def get_cpu_copy(self, indices, mamba_indices=None):
        self.export_locations(indices)
        return super().get_cpu_copy(indices, mamba_indices)

    def load_cpu_copy(self, cache_cpu, indices, mamba_indices=None):
        super().load_cpu_copy(cache_cpu, indices, mamba_indices)
        self.import_locations(indices)

    def get_kv_size_bytes(self):
        k, v = super().get_kv_size_bytes()
        # During the base constructor the slab attributes are not bound yet.
        if hasattr(self, "slab_key"):
            k += self.slab_key.numel() * self.slab_key.element_size()
            v += self.slab_value.numel() * self.slab_value.element_size()
        return k, v


class Q35rtHybridKVPool(HybridLinearKVPool):
    """Use native PD ownership with an explicit BF16 convolution receive stage."""

    def __init__(self, *, draft_pool, pd_staging=False, **kwargs):
        super().__init__(**kwargs)
        self._draft_pool = draft_pool
        self._conv_receive = None
        self._direct_pd_slab = False
        self._ssm_layout = recurrent_state_layout()
        if pd_staging:
            conv = self.mamba_pool.mamba_cache.conv[0]
            self._conv_receive = torch.zeros(
                conv.shape, dtype=torch.bfloat16, device=conv.device
            )

    def get_state_buf_infos(self):
        ptrs, lengths, items = super().get_state_buf_infos()
        if self._conv_receive is not None:
            conv = self.mamba_pool.mamba_cache.conv[0]
            replacements = {
                conv[i].data_ptr(): self._conv_receive[i] for i in range(conv.shape[0])
            }
            for i, pointer in enumerate(ptrs):
                if pointer in replacements:
                    plane = replacements[pointer]
                    ptrs[i], lengths[i], items[i] = (
                        plane.data_ptr(),
                        plane.nbytes,
                        plane[0].nbytes,
                    )
        return ptrs, lengths, items

    def q35rt_bind_pd_manager(self, manager):
        value = os.environ.get("SGLANG_Q35RT_DIRECT_PD_SLAB", "0")
        if value not in ("0", "1"):
            raise ValueError("SGLANG_Q35RT_DIRECT_PD_SLAB must be 0 or 1")
        if value == "1":
            manager.q35rt_slab_targets = (
                self.full_kv_pool.slab_key, self.full_kv_pool.slab_value,
                self._draft_pool.slab_key, self._draft_pool.slab_value,
            )
            self._direct_pd_slab = True

    def q35rt_import_received(self, req, req_pool):
        # Native receiver wrote only delta KV. Cached prefix slabs remain live;
        # their canonical transport buffers need not have been refreshed.
        start = req.cache_protected_len
        end = len(req.origin_input_ids)
        logger.info(
            "q35rt PD import: rid=%s reused_tokens=%s new_tokens=%s",
            req.rid,
            start,
            end - start,
        )
        locations = req_pool.req_to_token[req.req_pool_idx, start:end].to(torch.int64)
        if os.environ.get("Q35_PD_VERIFY_DIR"):
            from sglang.srt.disaggregation.common.q35rt_staging import verify_received

            if self._direct_pd_slab:
                # Debug-only independent export through the validated old
                # layout converter; compare the actual slabs with P packets.
                self.full_kv_pool.export_locations(locations)
                self._draft_pool.export_locations(locations)
            verify_received(
                req,
                locations,
                self.full_kv_pool.k_buffer
                + self.full_kv_pool.v_buffer
                + self._draft_pool.k_buffer
                + self._draft_pool.v_buffer,
            )
        if not self._direct_pd_slab:
            self.full_kv_pool.import_locations(locations)
            self._draft_pool.import_locations(locations)
        if self._conv_receive is not None or self._ssm_layout == "chunked_kv":
            slots = req_pool.translate_mamba_indices(
                req_pool.req_index_to_mamba_index_mapping[
                    req.req_pool_idx : req.req_pool_idx + 1
                ]
            )
        if self._conv_receive is not None:
            if os.environ.get("Q35_PD_VERIFY_DIR"):
                from sglang.srt.disaggregation.common.q35rt_staging import verify_state

                verify_state(
                    req,
                    self.mamba_pool.mamba_cache.temporal[:, slots[0]],
                    self._conv_receive[:, slots[0]],
                )
            self.mamba_pool.mamba_cache.conv[0][:, slots] = self._conv_receive[
                :, slots
            ].float()
        if self._ssm_layout == "chunked_kv":
            from q35rt.cache_ops import transpose_native_ssm_inplace

            # NIXL always receives the standard VK wire format. Cache COW,
            # checkpoints and offload copy whole opaque slots; only q35rt
            # compute reads the chunked physical encoding after admission.
            transpose_native_ssm_inplace(self.mamba_pool.mamba_cache.temporal, slots)
        # PD supplies the terminal recurrent state, superseding the deferred
        # cold clear / matched-prefix COW used by local extend forwards.
        req.mamba_needs_clear = False
        req.mamba_cow_src_index = None


class Q35rtForCausalLM(nn.Module):
    """Opt-in leaf compute model driven by the native SGLang worker."""

    q35rt_direct_logits = True
    cuda_graph_reads_req_to_token = True
    q35rt_build_attention_backend = staticmethod(build_q35rt_attention_backend)
    q35rt_prefill_materializes_context = True
    # Shared native target/draft page ownership, explicit accepted-state
    # checkpoints and delta-only PD import satisfy UnifiedRadixCache's contract.
    q35rt_pd_radix_cache = True
    # Target + draft, canonical transfer + compute. Logical scratch is a
    # separate fixed reservation: context length can exceed the pool budget.
    q35rt_kv_bytes_per_token = 2 * (10 * 2 * 256 * 4 + 6 * 8 * 128 * 4)

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        from q35rt.framework import FrameworkEngine

        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        if any(os.environ.get(name) for name in ("VDEFS", "Q35_SO", "VT_MAXREG")):
            raise ValueError(
                "q35rt serving requires the pinned build without experimental overrides"
            )
        if os.environ.get("Q35_DETERMINISTIC", "0") not in ("0", "1"):
            raise ValueError("Q35_DETERMINISTIC must be exactly 0 or 1")
        text_config = getattr(config, "text_config", config)
        if (
            text_config.hidden_size,
            text_config.num_hidden_layers,
            text_config.vocab_size,
        ) != (2048, 40, 248320):
            raise ValueError("q35rt requires the Qwen3.5-35B-A3B model geometry")
        if (
            args.tp_size != 1
            or args.pp_size != 1
            or args.max_running_requests != 1
            or args.speculative_algorithm != "DFLASH"
            or args.speculative_num_draft_tokens != 8
        ):
            raise ValueError(
                "q35rt requires TP1/PP1, max-running-requests=1 and DFLASH block8"
            )
        if torch.cuda.get_device_capability() != (10, 0):
            raise ValueError("q35rt serving currently supports B200 (sm100)")
        if args.speculative_draft_window_size is not None:
            raise ValueError("q35rt requires the native shared draft cache mapping")
        if args.enable_unified_memory or args.enable_lora:
            raise ValueError(
                "q35rt does not yet support unified-memory relocation or LoRA"
            )
        if (
            args.enable_hierarchical_cache
            or args.radix_cache_backend is not None
            or args.enable_memory_saver
            or args.enable_linear_replayssm
        ):
            raise ValueError(
                "q35rt requires the native device-resident cache and exact Mamba state"
            )
        if not args.language_only:
            raise ValueError("q35rt currently requires --language-only")
        if args.enable_int8_mamba_checkpoint or args.enable_page_major_kv_layout:
            raise ValueError(
                "q35rt currently requires exact native Mamba checkpoints and standard page layout"
            )
        if (
            args.quantization is not None
            or args.speculative_draft_model_quantization is not None
        ):
            raise ValueError(
                "q35rt packs its own FP8 weights; external quantization methods are unsupported"
            )
        if args.is_embedding or args.enable_return_hidden_states:
            raise ValueError(
                "q35rt currently supports text generation without hidden-state export"
            )
        if args.page_size < 64 or args.page_size % 64:
            raise ValueError("q35rt requires page-size to be a multiple of 64")
        self.config = config
        self.quant_config = None
        self.start_layer, self.end_layer = 0, 40
        self.layers = nn.ModuleList()
        self.engine = FrameworkEngine(
            args.model_path, args.speculative_draft_model_path, max_len=64, T=8
        )
        self._native_weights_loaded = False
        with open(self.engine.vext.__file__, "rb") as module:
            module_sha = hashlib.sha256(module.read()).hexdigest()
        logger.info(
            "q35rt ready: framework_abi=%s deterministic=%s module_sha256=%s",
            self.engine.vext.framework_forward_abi(),
            self.engine.deterministic,
            module_sha,
        )
        _ENGINES[torch.cuda.current_device()] = self.engine
        self.embed_tokens = nn.Embedding.from_pretrained(self.engine.embed, freeze=True)
        self.adapter = None
        self._draft_pool = None
        self._context_length = args.context_length or getattr(
            config, "max_position_embeddings", 262144
        )
        self.q35rt_fixed_cache_bytes = (self._context_length + 32) * (
            64 * 4 + 128 * 4 + 4
        )
        self._pd_staging = args.disaggregation_mode == "decode"
        self.q35rt_pd_stage_bytes_per_slot = (
            30 * 8192 * 3 * 2 if self._pd_staging else 0
        )

    def load_weights(self, weights):
        # The leaf loader above packs the pinned target/draft directly. The
        # supplied iterator is deliberately left lazy, avoiding a second load.
        if self._native_weights_loaded:
            raise RuntimeError(
                "q35rt weight updates require restarting the P/D deployment"
            )
        self._native_weights_loaded = True
        return set()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_dflash_layers_to_capture(self, layer_ids):
        from q35rt.model_vt import CAP_LAYERS

        if list(layer_ids) != CAP_LAYERS:
            raise ValueError("q35rt requires the trained DFlash capture layers")

    def q35rt_build_kv_pool(self, kvc, sizes, req_pool):
        if kvc.kv_cache_dtype != torch.bfloat16:
            raise ValueError("q35rt requires BF16 KV storage")
        if kvc.post_capture_kv_active:
            raise ValueError("q35rt requires fixed KV backing before graph capture")
        capacity = sizes.max_total_num_tokens + kvc.pool_page_size
        device = self.engine.dev
        key = torch.zeros((10, 2, 4, capacity, 64), dtype=torch.bfloat16, device=device)
        value = torch.zeros_like(key)
        draft_key = torch.zeros(
            (6, 8, 2, capacity, 64), dtype=torch.bfloat16, device=device
        )
        draft_value = torch.zeros_like(draft_key)
        self.engine.bind_kv_storage(
            key, value, draft_key, draft_value, self._context_length + 16
        )
        common = dict(
            size=sizes.max_total_num_tokens,
            page_size=kvc.pool_page_size,
            dtype=torch.bfloat16,
            device=kvc.device,
            enable_memory_saver=False,
            enable_kv_cache_copy=True,
        )
        full = Q35rtNativeKVPool(
            key, value, layer_num=10, head_num=2, head_dim=256, **common
        )
        self._draft_pool = Q35rtNativeKVPool(
            draft_key, draft_value, layer_num=6, head_num=8, head_dim=128, **common
        )
        self.engine._sglang_draft_pool = self._draft_pool
        self.adapter = Q35rtComputeAdapter(
            self.engine, req_pool, page_size=kvc.pool_page_size
        )
        return Q35rtHybridKVPool(
            size=sizes.max_total_num_tokens,
            dtype=torch.bfloat16,
            page_size=kvc.pool_page_size,
            head_num=2,
            head_dim=256,
            full_attention_layer_ids=kvc.mambaish_config.full_attention_layer_ids,
            device=kvc.device,
            mamba_pool=req_pool.mamba_pool,
            full_kv_pool=full,
            enable_kv_cache_copy=True,
            draft_pool=self._draft_pool,
            pd_staging=self._pd_staging,
        )

    def q35rt_commit(self, batch, commit_lens, bonus, prefix_lens):
        slot, count = None, None
        if batch.mamba_track_indices is not None:
            from sglang.srt.runtime_context import get_exec

            interval = get_exec().mamba.mamba_track_interval
            end = prefix_lens + commit_lens.to(prefix_lens.dtype)
            relative = end // interval * interval - prefix_lens
            count = torch.where(
                (relative > 0) & (relative <= commit_lens), relative, -1
            ).to(torch.int64)
            slot = self.adapter.req_to_token_pool.translate_mamba_indices(
                batch.mamba_track_indices
            )
        self.adapter.commit(commit_lens, bonus, batch.req_pool_indices, slot, count)
        # Context must be materialized before native radix publication.
        self.adapter.forward_context(batch.req_pool_indices)

    def _debug_state(self, label, batch, length):
        import hashlib
        import json

        from q35rt.cache_ops import convert_ssm_layout, gather

        pool = self.adapter.req_to_token_pool
        req = int(batch.req_pool_indices[0])
        slot = int(pool.req_index_to_mamba_index_mapping[req])
        state = pool.mamba_pool.mamba_cache
        tensors = {"ssm": convert_ssm_layout(state.temporal[:, slot],
                    self.engine.recurrent_state_layout, "vk"), "conv": state.conv[0][:, slot]}
        if length:
            loc = pool.req_to_token[req, :length].to(torch.int64)
            tensors.update(
                key=gather(self.engine.t_kc, loc),
                value=gather(self.engine.t_vc, loc),
                draft_key=gather(self.engine.d_kc, loc),
                draft_value=gather(self.engine.d_vc, loc),
            )
        row = dict(label=label, length=length, request=req, slot=slot)
        for name, value in tensors.items():
            row[name] = hashlib.sha256(
                value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
        if batch.mamba_track_indices is not None:
            tracked = int(batch.mamba_track_indices[0])
            row["track_slot"] = tracked
            for name, value in (
                ("track_ssm", convert_ssm_layout(state.temporal[:, tracked],
                    self.engine.recurrent_state_layout, "vk")),
                ("track_conv", state.conv[0][:, tracked]),
            ):
                row[name] = hashlib.sha256(
                    value.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                ).hexdigest()
        print("[q35rt-state] " + json.dumps(row), flush=True)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        if forward_batch.forward_mode.is_idle():
            return LogitsProcessorOutput(next_token_logits=self.engine.t_logits[:0])
        if forward_batch.forward_mode.is_target_verify():
            return self.adapter.forward_target(forward_batch)
        if not forward_batch.forward_mode.is_extend() or forward_batch.batch_size != 1:
            raise ValueError(
                "q35rt supports BS1 native extend and DFLASH target verify"
            )
        # Slow local prefill supports native smoke/recovery. Production PD uses
        # the native CP2 prefill worker and transfers its complete cache.
        self.adapter._bind_request(forward_batch.req_pool_indices)
        # Native MoE planning may pad the shared input/position buffers to a
        # tile boundary. Only the real extension belongs to this request's
        # recurrent state, cache and final logits.
        n = forward_batch.extend_seq_lens_cpu[0]
        if n > input_ids.numel():
            raise ValueError("q35rt extend metadata exceeds the input buffer")
        if n == 0:
            raise ValueError("q35rt extend requires at least one uncached token")
        debug_state = os.environ.get("SGLANG_Q35RT_DEBUG_STATE", "0") == "1"
        if debug_state:
            import json

            print(
                "[q35rt-input] "
                + json.dumps(
                    dict(
                        n=n,
                        buffer=input_ids.numel(),
                        prefix=forward_batch.extend_prefix_lens_cpu,
                        positions=positions[:4].tolist() + positions[-4:].tolist(),
                        tail_ids=input_ids[-8:].tolist(),
                        track=(
                            None
                            if forward_batch.mamba_track_seqlens is None
                            else forward_batch.mamba_track_seqlens.tolist()
                        ),
                        mask=(
                            None
                            if forward_batch.mamba_track_mask is None
                            else forward_batch.mamba_track_mask.tolist()
                        ),
                    )
                ),
                flush=True,
            )
            self._debug_state(
                "extend-start", forward_batch, forward_batch.extend_prefix_lens_cpu[0]
            )
        for start in range(0, n, 8):
            count = min(8, n - start)
            block = torch.zeros(8, dtype=torch.int64, device=input_ids.device)
            block[:count].copy_(input_ids[start : start + count])
            if count < 8:
                # The fixed-width target writes KV for every row. Unallocated
                # tail entries may still refer to a previous use of this req
                # slot, including a live cached prefix. Give padded rows unique
                # locations in the allocator's reserved page zero. The native
                # worker replaces these entries when it reserves verify rows.
                logical = positions[start] + torch.arange(
                    count, 8, device=input_ids.device
                )
                padding = torch.arange(
                    8 - count, dtype=torch.int32, device=input_ids.device
                )
                self.adapter.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices[:, None], logical[None, :]
                ] = padding[None, :]
            result = self.engine.target_forward(block, positions[start : start + 1])
            committed = torch.tensor(
                [count], dtype=torch.int32, device=input_ids.device
            )
            bonus = torch.zeros(1, dtype=torch.int64, device=input_ids.device)
            slot, track_count = None, None
            if forward_batch.mamba_track_indices is not None:
                from sglang.srt.server_args import get_global_server_args

                chunk = get_global_server_args().mamba_cache_chunk_size
                prefix = forward_batch.extend_prefix_lens
                target = (
                    prefix
                    + (forward_batch.mamba_track_seqlens - prefix) // chunk * chunk
                )
                relative = target - positions[start : start + 1]
                track_count = torch.where(
                    forward_batch.mamba_track_mask
                    & (relative > 0)
                    & (relative <= count),
                    relative,
                    -1,
                ).to(torch.int64)
                slot = forward_batch.mamba_track_indices
            self.adapter.commit(
                committed, bonus, forward_batch.req_pool_indices, slot, track_count
            )
            self.adapter.forward_context(forward_batch.req_pool_indices)
            if debug_state and int(positions[start]) + count == 576:
                self._debug_state("active-at-576", forward_batch, 576)
        if debug_state:
            self._debug_state(
                "extend-end", forward_batch, forward_batch.extend_prefix_lens_cpu[0] + n
            )
        return LogitsProcessorOutput(
            next_token_logits=result.logits[count - 1 : count],
            hidden_states=result.hidden_states[count - 1 : count],
        )


class Q35rtDFlashDraftModel(nn.Module):
    q35rt_direct_logits = True
    cuda_graph_reads_req_to_token = True
    q35rt_build_attention_backend = staticmethod(build_q35rt_attention_backend)
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = None
        self.engine = _ENGINES[torch.cuda.current_device()]
        self.layers = nn.ModuleList()
        self.start_layer, self.end_layer = 0, 6
        self.block_size = 8
        self.adapter = None
        self._native_weights_loaded = False

    def load_weights(self, weights):
        if self._native_weights_loaded:
            raise RuntimeError(
                "q35rt draft weight updates require restarting the P/D deployment"
            )
        self._native_weights_loaded = True
        return set()

    def q35rt_build_kv_pool(self, kvc, sizes, req_pool):
        self.adapter = Q35rtComputeAdapter(
            self.engine, req_pool, page_size=kvc.pool_page_size
        )
        return self.engine._sglang_draft_pool

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        if forward_batch.forward_mode.is_idle():
            return LogitsProcessorOutput(next_token_logits=self.engine.draft_logits[:0])
        return self.adapter.forward_query(forward_batch)


class Q35rtSlabKVCache(KVCache):
    """Expose backend-specific slab storage to the native page allocator.

    The engine borrows these same buffers; transfers/imports use canonical KV.
    No separate service cache or per-step prefix gather is introduced.
    """

    def __init__(self, key, value, layer_ids, page_size=64):
        if key.shape != value.shape or key.ndim != 5 or key.shape[-1] != 64:
            raise ValueError("q35rt requires matching slab K/V tensors")
        if page_size < 64 or page_size % 64:
            raise ValueError("q35rt pages must contain whole 64-token groups")
        capacity = key.shape[3]
        if capacity % page_size:
            raise ValueError("physical KV capacity must be page aligned")
        super().__init__(
            size=capacity - page_size,
            page_size=page_size,
            dtype=key.dtype,
            layer_num=len(layer_ids),
            device=str(key.device),
            enable_memory_saver=False,
        )
        self.key, self.value = key, value
        self.layer_ids = list(layer_ids)
        self.layer_index = {layer: i for i, layer in enumerate(layer_ids)}
        self.head_num = key.shape[1]
        self.head_dim = key.shape[2] * 64

    def get_key_buffer(self, layer_id):
        return self.key[self.layer_index[layer_id]]

    def get_value_buffer(self, layer_id):
        return self.value[self.layer_index[layer_id]]

    def get_kv_buffer(self, layer_id):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(self, layer, loc, cache_k, cache_v, **kwargs):
        from q35rt.cache_ops import scatter

        i = self.layer_index[layer.layer_id]
        scatter(self.key[i : i + 1], loc, cache_k.unsqueeze(0))
        scatter(self.value[i : i + 1], loc, cache_v.unsqueeze(0))

    def get_kv_size_bytes(self):
        return tuple(t.numel() * t.element_size() for t in (self.key, self.value))

    def get_contiguous_buf_infos(self):
        pointers, lengths, item_lengths = [], [], []
        for tensor in (self.key, self.value):
            planes = tensor.reshape(-1, tensor.shape[3], 64)
            for plane in planes:
                pointers.append(plane.data_ptr())
                lengths.append(plane.numel() * plane.element_size())
                item_lengths.append(64 * plane.element_size())
        return pointers, lengths, item_lengths


class Q35rtComputeAdapter:
    def __init__(
        self, engine, req_to_token_pool, draft_req_to_token_pool=None, page_size=64
    ):
        self.engine = engine
        self.req_to_token_pool = req_to_token_pool
        self.draft_req_to_token_pool = draft_req_to_token_pool or req_to_token_pool
        self.page_size = page_size
        if hasattr(req_to_token_pool, "mamba_pool"):
            state = req_to_token_pool.mamba_pool.mamba_cache
            if len(state.conv) != 1:
                raise ValueError("q35rt GDN requires one combined convolution state")
            engine.bind_recurrent_pool(
                state.temporal,
                state.conv[0],
                req_to_token_pool.req_index_to_mamba_index_mapping,
                state_layout=recurrent_state_layout(),
            )

    def _bind_request(self, req_pool_indices):
        self.engine.bind_cache_maps(
            self.req_to_token_pool.req_to_token,
            req_pool_indices,
            self.draft_req_to_token_pool.req_to_token,
            self.page_size,
        )

    def forward_target(self, forward_batch):
        if forward_batch.batch_size != 1:
            raise ValueError("q35rt initial backend supports request BS1")
        if not forward_batch.forward_mode.is_target_verify():
            raise ValueError("q35rt initial compute path is target verify only")
        self._bind_request(forward_batch.req_pool_indices)
        result = self.engine.target_forward(
            forward_batch.input_ids, forward_batch.positions[:1]
        )
        return LogitsProcessorOutput(
            next_token_logits=result.logits,
            hidden_states=result.hidden_states,
        )

    def commit(
        self,
        accepted_input_count,
        bonus,
        req_pool_indices=None,
        checkpoint_slot=None,
        checkpoint_count=None,
    ):
        # Replays do not execute Python forward: never assume its last borrowed
        # request-index pointer still identifies the active ScheduleBatch.
        if req_pool_indices is not None:
            self._bind_request(req_pool_indices)
        self.engine.commit(
            accepted_input_count, bonus, checkpoint_slot, checkpoint_count
        )

    def forward_context(self, req_pool_indices=None):
        if req_pool_indices is not None:
            self._bind_request(req_pool_indices)
        self.engine.context_forward()

    def forward_query(self, forward_batch):
        if (
            forward_batch.batch_size != 1
            or not forward_batch.forward_mode.is_target_verify()
        ):
            raise ValueError("q35rt draft query requires native BS1/T8 TARGET_VERIFY")
        self._bind_request(forward_batch.req_pool_indices)
        result = self.engine.query_forward(
            forward_batch.input_ids, forward_batch.positions[:1]
        )
        return LogitsProcessorOutput(next_token_logits=result.logits,
            customized_info={"q35rt_draft_top1": result.top1} if result.top1 is not None else None)

    def forward_draft(self):
        result = self.engine.draft_forward()
        return LogitsProcessorOutput(next_token_logits=result.logits,
            customized_info={"q35rt_draft_top1": result.top1} if result.top1 is not None else None)
