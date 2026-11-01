"""Seed native DFlash overlap state after the PD cache handoff completes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2


def build_dflash_disagg_draft_input(
    batch: ScheduleBatch,
    server_args: ServerArgs,
    last_tokens_tensor: torch.Tensor,
    future_map: FutureMap,
) -> DFlashDraftInputV2:
    # Prompt context KV was materialized by P and transferred by the native
    # receiver. The first D step starts from P's sampled, unprocessed token.
    spec_info = make_draft_input_v2(
        bonus_tokens=last_tokens_tensor, new_seq_lens=batch.seq_lens
    )
    if batch.enable_overlap:
        spec_info.future_indices = batch.req_pool_indices
        future_map.publish(spec_info.future_indices, batch.seq_lens)
        future_map.stash(
            spec_info.future_indices, RelayPayload.from_draft_input(spec_info)
        )
    return spec_info
