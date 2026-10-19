import threading
import unittest
from contextlib import nullcontext
from http import HTTPStatus
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from flexkv.common.request import KVResponseStatus
from flexkv.kvtask import KVTaskEngine, TaskStatus, TaskType
from sglang.srt.managers.kv_hints import decode_kv_hints_envelope
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage.flexkv import _flexkv_factory
from sglang.srt.mem_cache.storage.flexkv.flexkv_connector import FlexKVConnector
from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import FlexKVRadixCache


class TestFlexKVRefreshHint(unittest.TestCase):
    def test_only_supported_action_is_consumed_once(self):
        for kind, version, expected in (
            ("flexkv.refresh_on_miss", "1", True),
            ("other.action", "1", False),
            ("flexkv.refresh_on_miss", "2", False),
        ):
            req = SimpleNamespace(
                kv_hints=decode_kv_hints_envelope(
                    {
                        "protocol_version": "1",
                        "message_id": "retry",
                        "actions": [
                            {
                                "action_id": "refresh",
                                "action_type": kind,
                                "action_version": version,
                            }
                        ],
                    }
                )
            )
            self.assertEqual(FlexKVRadixCache._consume_refresh_hint(req), expected)
            self.assertFalse(FlexKVRadixCache._consume_refresh_hint(req))
        self.assertFalse(FlexKVRadixCache._consume_refresh_hint(None))
        self.assertFalse(FlexKVRadixCache._consume_refresh_hint(SimpleNamespace()))

    def test_connector_only_forwards_opted_in_refresh(self):
        connector = FlexKVConnector.__new__(FlexKVConnector)
        connector.page_size = 1
        connector._sync_ctx = SimpleNamespace(is_sync_leader=True, needs_sync=False)
        connector.kv_manager = mock.Mock()
        connector.kv_manager.get_match.return_value = (42, np.zeros(4, dtype=bool))
        for enabled in (False, True):
            connector.lookup_kv(
                [1, 2, 3, 4], torch.ones(4, dtype=torch.bool), refresh_on_miss=enabled
            )
            options = connector.kv_manager.get_match.call_args.kwargs
            self.assertEqual(options.get("refresh_on_miss", False), enabled)
            self.assertEqual("refresh_on_miss" in options, enabled)


class TestFlexKVStoreRankConsistency(unittest.TestCase):
    def test_registration_distinguishes_physical_and_global_kv_heads(self):
        for physical_heads, global_heads in ((8, 32), (1, 4), (1, 1)):
            with self.subTest(physical_heads=physical_heads, global_heads=global_heads):
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
        self.completed_task_ids = None

    def put_match(self, *, token_ids, token_mask):
        return 41, np.ones(len(token_ids), dtype=np.bool_)

    def launch(self, **kwargs):
        self.launched.append(kwargs)

    def try_wait(self, *, task_ids):
        completed = (
            task_ids if self.completed_task_ids is None else self.completed_task_ids
        )
        return {
            task_id: SimpleNamespace(status=self.status)
            for task_id in task_ids
            if task_id in completed
        }

    def wait(self, task_ids, *, timeout, completely):
        assert completely
        return {
            task_id: SimpleNamespace(
                status=(
                    self.status
                    if self.completed_task_ids is None
                    or task_id in self.completed_task_ids
                    else KVResponseStatus.TIMEOUT
                )
            )
            for task_id in task_ids
        }


def _connector(*, leader: bool, fanout: _Fanout, manager=None):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._sync_ctx = _SyncContext(leader=leader, fanout=fanout)
    connector.kv_manager = manager
    connector._inflight_stores = {}
    connector._inflight_loads = {}
    connector._launched_load_tids = []
    connector._pending_lookups = {}
    connector._ongoing_prefetches = {}
    connector._completed_layerwise = []
    connector.layer_done_counter = None
    connector._send_pp_put_meta = lambda *_args: None
    connector._send_slot_mapping_to_remote = lambda *_args: None
    return connector


class TestFlexKVStoreOwnership(unittest.TestCase):
    def test_real_backend_zero_timeout_wait_requires_complete_graph(self):
        backend = KVTaskEngine.__new__(KVTaskEngine)
        task = SimpleNamespace(
            status=TaskStatus.RUNNING,
            task_end_op_finished=True,
            task_type=TaskType.PUT,
            transfer_failed=False,
            return_mask=None,
        )
        task.is_completed = lambda: task.status == TaskStatus.COMPLETED
        backend.tasks = {41: task}
        backend._update_tasks = lambda **_kwargs: None
        backend._process_empty_graph = lambda _task_id: None
        backend._release_task = mock.Mock()
        self.assertTrue(backend.check_completed(41, completely=False))
        self.assertEqual(
            backend._wait_impl([41], timeout=0.0, completely=True)[41].status,
            KVResponseStatus.TIMEOUT,
        )
        backend._release_task.assert_not_called()
        task.status = TaskStatus.COMPLETED
        self.assertEqual(
            backend._wait_impl([41], timeout=0.0, completely=True)[41].status,
            KVResponseStatus.SUCCESS,
        )
        backend._release_task.assert_called_once_with(41)

    def test_poll_does_not_accept_an_early_task_end_acknowledgement(self):
        manager = mock.MagicMock()
        manager.try_wait.return_value = {
            41: SimpleNamespace(status=KVResponseStatus.SUCCESS)
        }
        manager.wait.return_value = {
            41: SimpleNamespace(status=KVResponseStatus.TIMEOUT)
        }
        connector = _connector(leader=True, fanout=_Fanout(), manager=manager)
        handle = CacheRequestHandle("store", 0)
        connector._inflight_stores[handle] = 41
        self.assertEqual(connector.check_completed_stores(), [])
        self.assertEqual(connector._inflight_stores, {handle: 41})
        manager.wait.assert_called_once_with([41], timeout=0.0, completely=True)
        manager.try_wait.assert_not_called()

    def test_aborted_prefetch_remains_tracked_until_reset_drains_it(self):
        manager = _KVManager()
        manager.reset = mock.Mock()
        manager.wait = mock.Mock(return_value={})
        connector = _connector(leader=True, fanout=_Fanout(), manager=manager)
        handle = CacheRequestHandle("prefetch", 0)
        connector._ongoing_prefetches[handle] = 41
        connector.cancel_prefetch(handle)
        with self.assertRaisesRegex(TimeoutError, "remains in flight"):
            connector.reset()
        manager.reset.assert_not_called()
        self.assertEqual(connector._ongoing_prefetches, {handle: 41})

    def test_incomplete_load_cannot_release_gpu_slots(self):
        fanout = _Fanout()
        manager = mock.MagicMock()
        manager.wait.return_value = {}
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        handle = CacheRequestHandle("load", 0)
        for connector in (leader, follower):
            connector._pending_lookups[handle] = 41
            connector._sync_ctx.should_send_slot_mapping_to_remote = False
            with self.assertRaisesRegex(TimeoutError, "slots must remain owned"):
                connector.retrieve_kv(handle, torch.tensor([1, 2]))
        self.assertEqual(leader._launched_load_tids, [41])
        manager.wait.assert_called_once_with([41], timeout=30.0, completely=True)

    def test_failed_load_returns_same_result_on_all_ranks(self):
        fanout = _Fanout()
        manager = _KVManager(KVResponseStatus.FAILED)
        manager.launch = mock.Mock()
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        handle = CacheRequestHandle("load", 0)
        for connector in (leader, follower):
            connector._pending_lookups[handle] = 41
            connector._sync_ctx.should_send_slot_mapping_to_remote = False
            self.assertEqual(connector.retrieve_kv(handle, torch.tensor([1, 2])), 0)
        self.assertEqual(leader._launched_load_tids, [])

    def test_pending_retract_store_keeps_ownership_on_every_rank(self):
        fanout = _Fanout()
        manager = _KVManager()
        manager.wait = mock.Mock(return_value={})
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        handle = CacheRequestHandle("req", 0)
        for connector in (leader, follower):
            connector._inflight_stores[handle] = 41
            with self.assertRaisesRegex(TimeoutError, "must remain owned"):
                connector.wait_store(handle, timeout=0.0)
            self.assertEqual(connector._inflight_stores, {handle: 41})

    def test_sleep_does_not_unmap_pending_device_io(self):
        manager = _KVManager()
        manager.wait = mock.Mock(return_value={})
        connector = _connector(leader=True, fanout=_Fanout(), manager=manager)
        connector.tp_client = mock.Mock()
        connector._launched_load_tids = [41]
        with self.assertRaisesRegex(TimeoutError, "remains in flight"):
            connector.before_device_sleep()
        connector.tp_client.suspend_gpu_mappings.assert_not_called()
        self.assertEqual(connector._launched_load_tids, [41])

    def test_local_reset_does_not_reset_backing_store(self):
        manager = _KVManager()
        manager.reset = mock.Mock()
        connector = _connector(leader=True, fanout=_Fanout(), manager=manager)
        connector.reset(clear_store=False)
        manager.reset.assert_not_called()
        connector.reset()
        manager.reset.assert_called_once()

    def test_abort_cleanup_retains_store_pin(self):
        handle = CacheRequestHandle("req", 0)
        node = object()
        cache = FlexKVRadixCache.__new__(FlexKVRadixCache)
        cache._load_markers = {handle: object()}
        cache._inflight_store_nodes = {handle: node}
        cache._node_lock = threading.Lock()
        cache.flexkv_connector = mock.Mock()
        cache.dec_lock_ref = mock.Mock()
        cache.release_aborted_request(handle)
        self.assertEqual(cache._inflight_store_nodes, {handle: node})
        cache.dec_lock_ref.assert_not_called()
        cache.flexkv_connector.release_pending.assert_called_once_with(handle)

    def test_store_and_completion_are_rank_consistent(self):
        fanout = _Fanout()
        manager = _KVManager()
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        token_ids = [1, 2, 3]
        kv_indices = torch.tensor([10, 11, 12])
        handle = CacheRequestHandle("req", 0)

        self.assertEqual(leader.store_kv(handle, token_ids, kv_indices), 41)
        self.assertEqual(follower.store_kv(handle, token_ids, kv_indices), 41)
        self.assertEqual(leader._inflight_stores, {handle: 41})
        self.assertEqual(follower._inflight_stores, {handle: 41})

        self.assertEqual(leader.check_completed_stores(), [handle])
        self.assertEqual(follower.check_completed_stores(), [handle])
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

    def test_completion_poll_keeps_unfinished_store(self):
        fanout = _Fanout()
        manager = _KVManager()
        manager.completed_task_ids = []
        leader = _connector(leader=True, fanout=fanout, manager=manager)
        follower = _connector(leader=False, fanout=fanout)
        leader._inflight_stores = {"req": 41}
        follower._inflight_stores = {"req": 41}

        self.assertEqual(leader.check_completed_stores(), [])
        self.assertEqual(follower.check_completed_stores(), [])
        self.assertEqual(leader._inflight_stores, {"req": 41})
        self.assertEqual(follower._inflight_stores, {"req": 41})

        manager.completed_task_ids = [41]
        self.assertEqual(leader.check_completed_stores(), ["req"])
        self.assertEqual(follower.check_completed_stores(), ["req"])
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
            kv=SimpleNamespace(req_pool_idx=0),
            cache_request_handle=CacheRequestHandle("req", 0),
            origin_input_ids=[1, 2],
            output_ids=[3, 4],
            owned_kv_len=lambda: 3,
        )

        with mock.patch("torch.cuda.stream", return_value=nullcontext()):
            self.assertTrue(cache.checkpoint_retracted_req(req, timeout=7.0))

        store_call = connector.store_kv.call_args.kwargs
        self.assertEqual(store_call["handle"], req.cache_request_handle)
        self.assertEqual(store_call["token_ids"], [1, 2, 3])
        self.assertEqual(store_call["kv_indices"].tolist(), [10, 11, 12])
        connector.wait_store.assert_called_once_with(
            req.cache_request_handle, timeout=7.0
        )

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
        self.assertEqual(FlexKVRadixCache._finished_store_event(req), "cancel_abort")

        req.finished_reason = reason(
            "abort", status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self.assertEqual(FlexKVRadixCache._finished_store_event(req), "error_abort")

    def test_abort_req_carries_checkpoint_intent(self):
        from sglang.srt.managers.io_struct import AbortReq

        self.assertFalse(AbortReq(rid="req").checkpoint_aborted_kv)
        self.assertTrue(
            AbortReq(rid="req", checkpoint_aborted_kv=True).checkpoint_aborted_kv
        )


if __name__ == "__main__":
    unittest.main()
