import unittest
from contextlib import nullcontext
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

try:
    import flexkv  # noqa: F401
    from flexkv.common.request import KVResponseStatus
except ImportError:
    flexkv = None
    KVResponseStatus = None

from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector
from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import FlexKVRadixCache


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
    def __init__(self, status=None):
        self.launched = []
        self.status = status or KVResponseStatus.SUCCESS

    def put_match(self, *, token_ids, token_mask):
        return 41, np.ones(len(token_ids), dtype=np.bool_)

    def launch(self, **kwargs):
        self.launched.append(kwargs)

    def try_wait(self, *, task_ids):
        return {task_id: object() for task_id in task_ids}

    def wait(self, task_ids, timeout, completely):
        return {
            task_id: SimpleNamespace(status=self.status) for task_id in task_ids
        }


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

    def test_wait_store_result_is_rank_consistent(self):
        for status, expected in (
            (KVResponseStatus.SUCCESS, True),
            (KVResponseStatus.FAILED, False),
        ):
            with self.subTest(status=status):
                fanout = _Fanout()
                manager = _KVManager(status=status)
                leader = _connector(leader=True, fanout=fanout, manager=manager)
                follower = _connector(leader=False, fanout=fanout)
                token_ids = [1, 2, 3]
                kv_indices = torch.tensor([10, 11, 12])

                leader.store_kv("req", token_ids, kv_indices)
                follower.store_kv("req", token_ids, kv_indices)
                self.assertIs(leader.wait_store("req"), expected)
                self.assertIs(follower.wait_store("req"), expected)
                self.assertEqual(leader._inflight_stores, {})
                self.assertEqual(follower._inflight_stores, {})

    def test_retract_checkpoint_stores_before_release(self):
        connector = SimpleNamespace(
            store_kv=mock.Mock(return_value=41),
            wait_store=mock.Mock(return_value=True),
        )
        cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
        cache.store_events = frozenset({"retract"})
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13]])
        )
        cache.store_stream = object()
        cache.flexkv_connector = connector
        req = SimpleNamespace(
            rid="req",
            req_pool_idx=0,
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            effective_kv_committed_len=lambda: 3,
        )

        with mock.patch("torch.cuda.stream", return_value=nullcontext()):
            self.assertTrue(cache.checkpoint_retracted_req(req, timeout=7.0))

        store_call = connector.store_kv.call_args.kwargs
        self.assertEqual(store_call["rid"], "req")
        self.assertEqual(store_call["token_ids"], [1, 2, 3])
        self.assertEqual(store_call["kv_indices"].tolist(), [10, 11, 12])
        connector.wait_store.assert_called_once_with("req", timeout=7.0)

    def test_hybrid_ssm_is_rejected_before_cache_construction(self):
        ctx = SimpleNamespace(is_hybrid_ssm=True)
        with self.assertRaisesRegex(ValueError, "hybrid SSM/GatedDeltaNet"):
            _flexkv_factory(ctx)

    def test_finished_store_event_classification(self):
        def reason(kind, status_code=None):
            return SimpleNamespace(
                to_json=lambda: {"type": kind, "status_code": status_code}
            )

        req = SimpleNamespace(
            finished_reason=reason("stop"), checkpoint_aborted_kv=False
        )
        self.assertEqual(FlexKVRadixCache._finished_store_event(req), "finish")

        req.finished_reason = reason("abort")
        req.checkpoint_aborted_kv = True
        self.assertEqual(
            FlexKVRadixCache._finished_store_event(req), "checkpoint_abort"
        )

        req.checkpoint_aborted_kv = False
        self.assertEqual(
            FlexKVRadixCache._finished_store_event(req), "cancel_abort"
        )

        req.finished_reason = reason(
            "abort", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self.assertEqual(
            FlexKVRadixCache._finished_store_event(req), "error_abort"
        )

    def test_abort_req_carries_checkpoint_intent(self):
        from sglang.srt.managers.io_struct import AbortReq

        self.assertFalse(AbortReq(rid="req").checkpoint_aborted_kv)
        self.assertTrue(
            AbortReq(rid="req", checkpoint_aborted_kv=True).checkpoint_aborted_kv
        )


if __name__ == "__main__":
    unittest.main()
