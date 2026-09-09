"""Opt-in, low-overhead scheduler timeline recording for offline analysis."""

from __future__ import annotations

import atexit
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Optional


class RequestTimelineRecorder:
    """Write one compact JSONL record per physical scheduler batch.

    The recorder is deliberately independent of SGLang internals. It reads only
    stable batch/request attributes and is a no-op unless an output directory is
    configured through ``SGLANG_REQUEST_TIMELINE_DIR``.
    """

    def __init__(
        self,
        *,
        output_dir: Optional[str],
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        gpu_id: int,
    ) -> None:
        self._file = None
        self._pending: dict[int, dict[str, Any]] = {}
        self._last_launch_iter: Optional[int] = None
        self._records_since_flush = 0
        self._flush_every = max(
            1, int(os.getenv("SGLANG_REQUEST_TIMELINE_FLUSH_EVERY", "1000"))
        )

        enabled = os.getenv("SGLANG_REQUEST_TIMELINE_TRACE", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not enabled or not output_dir or tp_rank != 0 or pp_rank != 0:
            return

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        hostname = socket.gethostname()
        rank_label = "none" if dp_rank is None else str(dp_rank)
        path = root / (
            f"engine-{hostname}-pid{os.getpid()}-dp{rank_label}-gpu{gpu_id}.jsonl"
        )
        self._file = path.open("a", encoding="utf-8", buffering=1024 * 1024)
        now_wall_ns = time.time_ns()
        now_monotonic_ns = time.monotonic_ns()
        self._write(
            {
                "phase": "TRACE_CLOCK_ANCHOR",
                "wall_ns": now_wall_ns,
                "monotonic_ns": now_monotonic_ns,
                "hostname": hostname,
                "pid": os.getpid(),
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "dp_rank": dp_rank,
                "gpu_id": gpu_id,
                "node_rank": int(os.getenv("SGLANG_REQUEST_TIMELINE_NODE_RANK", "-1")),
                "role": os.getenv("SGLANG_REQUEST_TIMELINE_ROLE", "unknown"),
            }
        )
        atexit.register(self.close)

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._records_since_flush += 1
        if self._records_since_flush >= self._flush_every:
            self._file.flush()
            self._records_since_flush = 0

    @staticmethod
    def _safe_len(value: Any) -> int:
        try:
            return len(value)
        except (TypeError, AttributeError):
            return 0

    @classmethod
    def _snapshot_batch(
        cls, batch: Any, wall_ns: int, monotonic_ns: int
    ) -> dict[str, Any]:
        requests = list(getattr(batch, "reqs", ()) or ())
        mode = getattr(batch, "forward_mode", None)
        mode_name = getattr(mode, "name", str(mode))

        request_ids = []
        prompt_tokens = []
        output_tokens = []
        device_cached_tokens = []
        host_cached_tokens = []
        storage_hit_tokens = []
        matched_prefix_tokens = []
        extend_tokens = []
        for req in requests:
            request_ids.append(str(getattr(req, "rid", "")))
            prompt_tokens.append(cls._safe_len(getattr(req, "origin_input_ids", ())))
            output_tokens.append(cls._safe_len(getattr(req, "output_ids", ())))
            device_cached_tokens.append(
                cls._safe_len(getattr(req, "prefix_indices", ()))
            )
            host_cached_tokens.append(int(getattr(req, "host_hit_length", 0) or 0))
            storage_hit_tokens.append(
                int(getattr(req, "storage_hit_length", 0) or 0)
            )
            matched_prefix_tokens.append(
                int(getattr(req, "num_matched_prefix_tokens", 0) or 0)
            )
            extend_range = getattr(req, "extend_range", None)
            extend_tokens.append(int(getattr(extend_range, "length", 0) or 0))

        return {
            "phase": "ENGINE_STEP",
            "start_wall_ns": wall_ns,
            "start_monotonic_ns": monotonic_ns,
            "forward_iter": int(getattr(batch, "forward_iter", -1) or -1),
            "forward_mode": mode_name,
            "after_idle_gap": bool(getattr(batch, "after_idle_gap", False)),
            "request_ids": request_ids,
            "request_prompt_tokens": prompt_tokens,
            "request_output_tokens": output_tokens,
            "request_device_cached_tokens": device_cached_tokens,
            "request_host_cached_tokens": host_cached_tokens,
            "request_storage_hit_tokens": storage_hit_tokens,
            "request_matched_prefix_tokens": matched_prefix_tokens,
            "request_extend_tokens": extend_tokens,
            "batch_size": len(requests),
            "extend_num_tokens": int(getattr(batch, "extend_num_tokens", 0) or 0),
        }

    def on_batch_launch(self, batch: Any) -> None:
        if self._file is None:
            return
        wall_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        record = self._snapshot_batch(batch, wall_ns, monotonic_ns)
        previous = self._pending.get(self._last_launch_iter)
        if previous is not None:
            previous["next_launch_wall_ns"] = wall_ns
            previous["next_launch_monotonic_ns"] = monotonic_ns
        self._pending[record["forward_iter"]] = record
        self._last_launch_iter = record["forward_iter"]

    def on_batch_result(self, batch: Any) -> None:
        if self._file is None:
            return
        forward_iter = int(getattr(batch, "forward_iter", -1) or -1)
        record = self._pending.pop(forward_iter, None)
        if record is None:
            return
        self._emit_record(
            record,
            time.time_ns(),
            time.monotonic_ns(),
            "batch_result",
        )

    def flush_pending(self, reason: str) -> None:
        if self._file is None:
            return
        for record in self._pending.values():
            record = dict(record)
            record["phase"] = "ENGINE_STEP_INCOMPLETE"
            record["close_reason"] = reason
            self._write(record)
        self._pending.clear()
        self._file.flush()
        self._records_since_flush = 0

    def _emit_record(
        self,
        record: dict[str, Any],
        end_wall_ns: int,
        end_monotonic_ns: int,
        reason: str,
    ) -> None:
        record["result_wall_ns"] = end_wall_ns
        record["result_monotonic_ns"] = end_monotonic_ns
        interval_end_wall_ns = min(
            end_wall_ns, record.get("next_launch_wall_ns", end_wall_ns)
        )
        interval_end_monotonic_ns = min(
            end_monotonic_ns,
            record.get("next_launch_monotonic_ns", end_monotonic_ns),
        )
        record["end_wall_ns"] = max(record["start_wall_ns"], interval_end_wall_ns)
        record["end_monotonic_ns"] = max(
            record["start_monotonic_ns"], interval_end_monotonic_ns
        )
        record["interval_end_reason"] = (
            "next_batch_launch"
            if interval_end_wall_ns < end_wall_ns
            else "batch_result"
        )
        record["close_reason"] = reason
        self._write(record)

    def close(self) -> None:
        if self._file is None:
            return
        self.flush_pending("process_exit")
        self._file.close()
        self._file = None
