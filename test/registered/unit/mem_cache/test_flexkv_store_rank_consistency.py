import unittest
from types import SimpleNamespace

import numpy as np
import torch

try:
    import flexkv  # noqa: F401
except ImportError:
    flexkv = None

from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector


class _Fanout:
    payload = None


class _SyncContext:
    def __init__(self, *, leader: bool, fanout: _Fanout):
        self.is_sync_leader = leader
        self.is_pp_receiver = False
        self.needs_sync = True
        self._fanout = fanout

    def scatter(self, payload):
        if self.is_sync_leader:
            self._fanout.payload = payload
        return self._fanout.payload


class _KVManager:
    def __init__(self):
        self.launched = []

    def put_match(self, *, token_ids, token_mask):
        return 41, np.ones(len(token_ids), dtype=np.bool_)

    def launch(self, **kwargs):
        self.launched.append(kwargs)

    def try_wait(self, *, task_ids):
        return {task_id: object() for task_id in task_ids}


def _connector(*, leader: bool, fanout: _Fanout, manager=None):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._sync_ctx = _SyncContext(leader=leader, fanout=fanout)
    connector.kv_manager = manager
    connector._inflight_stores = {}
    connector._send_pp_put_meta = lambda *_args: None
    connector._send_slot_mapping_to_remote = lambda *_args: None
    return connector


@unittest.skipIf(flexkv is None, "FlexKV is not installed")
class TestFlexKVStoreRankConsistency(unittest.TestCase):
    def test_store_and_completion_are_rank_consistent(self):
        fanout = _Fanout()
        manager = _KVManager()
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        token_ids = [1, 2, 3]
        kv_indices = torch.tensor([10, 11, 12])

        self.assertEqual(leader.store_kv("req", token_ids, kv_indices), 41)
        self.assertEqual(follower.store_kv("req", token_ids, kv_indices), 41)
        self.assertEqual(leader._inflight_stores, {"req": 41})
        self.assertEqual(follower._inflight_stores, {"req": 41})

        self.assertEqual(leader.check_completed_stores(), ["req"])
        self.assertEqual(follower.check_completed_stores(), ["req"])
        self.assertEqual(leader._inflight_stores, {})
        self.assertEqual(follower._inflight_stores, {})

    def test_hybrid_ssm_is_rejected_before_cache_construction(self):
        ctx = SimpleNamespace(is_hybrid_ssm=True)
        with self.assertRaisesRegex(ValueError, "hybrid SSM/GatedDeltaNet"):
            _flexkv_factory(ctx)


if __name__ == "__main__":
    unittest.main()
