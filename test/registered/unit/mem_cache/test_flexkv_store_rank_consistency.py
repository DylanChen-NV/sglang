import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector


class TestFlexKVStoreRankConsistency(unittest.TestCase):
    def test_registration_distinguishes_physical_and_global_kv_heads(self):
        for physical_heads, global_heads in ((8, 32), (1, 4), (1, 1)):
            with self.subTest(
                physical_heads=physical_heads, global_heads=global_heads
            ):
                connector = FlexKVConnector.__new__(FlexKVConnector)
                connector.model_config = SimpleNamespace(
                    kv_dim=2, num_kv_heads=global_heads
                )
                connector.rank_info = SimpleNamespace(num_layers_per_pp_stage=2)
                connector.page_size = 16
                connector.tp_client = mock.MagicMock()
                connector._label = "test"
                kv_caches = [torch.empty(160, physical_heads, 128)] * 4

                connector._register_to_server(kv_caches)

                layout = connector.tp_client.register_to_server.call_args.kwargs[
                    "kv_layout"
                ]
                self.assertEqual(layout.num_head, physical_heads)
                self.assertEqual(layout.num_kv_heads, global_heads)


if __name__ == "__main__":
    unittest.main()
