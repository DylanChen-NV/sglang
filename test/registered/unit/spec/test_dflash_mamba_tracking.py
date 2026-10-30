"""Filtered DFlash batches must refresh native checkpoint destinations."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput


class TestDflashMambaTracking(unittest.TestCase):
    def test_verify_refreshes_slots_after_filter_and_ping_pong_swap(self):
        factory = torch.tensor
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            req_pool_indices=factory([1]),
            reqs=[SimpleNamespace(mamba_next_track_idx=1)],
            req_to_token_pool=SimpleNamespace(
                req_index_to_mamba_ping_pong_track_buffer_mapping=factory([[9, 10], [15, 16]])
            ),
            mamba_track_indices=factory([9]),
            mamba_track_mask=factory([True]),
            mamba_track_seqlens=factory([512]),
        )
        verify = DFlashVerifyInput(draft_token=factory([1] * 8),
                                  positions=factory(list(range(8))), draft_token_num=8)
        server_args = SimpleNamespace(enable_mamba_extra_buffer=lambda: True,
                                      enable_mamba_extra_buffer_lazy=lambda: False)
        expected = 16

        def unpinned_tensor(*args, **kwargs):
            # Exercise the real slot gather without requiring a CUDA driver.
            kwargs.pop("pin_memory", None)
            return factory(*args, **kwargs)

        def init_forward(*args, **kwargs):
            self.assertEqual(batch.mamba_track_indices.tolist(), [expected])
            self.assertIsNone(batch.mamba_track_mask)
            self.assertIsNone(batch.mamba_track_seqlens)
            return SimpleNamespace()

        runner = SimpleNamespace(decode_cuda_graph_runner=None)
        target = SimpleNamespace(model_runner=runner)
        with patch("sglang.srt.speculative.spec_utils.get_server_args", return_value=server_args), \
             patch("sglang.srt.managers.schedule_batch.torch.tensor", side_effect=unpinned_tensor), \
             patch.object(ForwardBatch, "init_new", side_effect=init_forward):
            verify.prepare_for_verify(batch, target, defer_metadata_init=True)
            # Another retained request and a ping-pong swap invalidate the old
            # destination even when shape and tensor dtype remain unchanged.
            batch.req_pool_indices = factory([0])
            batch.reqs = [SimpleNamespace(mamba_next_track_idx=0)]
            batch.mamba_track_mask = factory([True])
            batch.mamba_track_seqlens = factory([1024])
            expected = 9
            verify.prepare_for_verify(batch, target, defer_metadata_init=True)


if __name__ == "__main__":
    unittest.main()
