"""Opt-in native Q35rt PD cancellation ownership protocol.

A failed transport is not evidence that remote writes have stopped. The source
must drain posted handles and fence its producer before certifying quiescence.
The destination retains its native allocation until every source certifies it.
"""

import os
import threading
from dataclasses import dataclass, field

Q35RT_QUIESCENCE_V1 = 1


def protocol_version():
    flags = [os.environ.get(name, "0") for name in (
        "SGLANG_Q35RT_QUIESCENCE", "SGLANG_Q35RT_EARLY_TOKEN"
    )]
    if any(value not in ("0", "1") for value in flags):
        raise ValueError("Q35rt quiescence/early-token flags must be 0 or 1")
    return int(os.environ.get("SGLANG_Q35RT_PD", "0") == "1" and "1" in flags)


@dataclass
class _SourceRoom:
    active: int = 0
    cancelled: bool = False
    failed: bool = False
    finalizing: bool = False
    quiesced: bool = False
    released: bool = False
    peers: set = field(default_factory=set)
    notified: set = field(default_factory=set)


class SourceQuiescence:
    """Serialize chunk admission against abort, without GPU work on success."""

    def __init__(self, is_live, fence, notify):
        self._is_live = is_live
        self._fence = fence
        self._notify = notify
        self._lock = threading.Lock()
        self._rooms = {}

    def begin(self, room, peers):
        with self._lock:
            if not self._is_live(room):
                return False
            state = self._rooms.setdefault(room, _SourceRoom())
            if state.cancelled or state.released:
                return False
            state.peers.update(peers)
            state.active += 1
            return True

    def cancel(self, room, peers=(), *, failed=False):
        with self._lock:
            state = self._rooms.setdefault(room, _SourceRoom())
            state.cancelled = True
            state.failed |= failed
            state.peers.update(peers)
            state.released |= not self._is_live(room)
        self._settle(room)

    def cancelled(self, room):
        with self._lock:
            state = self._rooms.get(room)
            return state is not None and state.cancelled

    def finish(self, room):
        with self._lock:
            state = self._rooms[room]
            assert state.active > 0
            state.active -= 1
        self._settle(room)

    def can_release(self, room):
        with self._lock:
            state = self._rooms.get(room)
            return state is None or (state.active == 0 and (
                not state.cancelled or state.quiesced
            ))

    def release(self, room):
        # Native clear must remove transfer_infos before this call. A stale
        # queued chunk then fails is_live even after this guard is discarded.
        with self._lock:
            state = self._rooms.get(room)
            if state is not None:
                state.released = True
                self._discard(room, state)

    def _discard(self, room, state):
        if state.released and state.active == 0 and not state.finalizing and (
            not state.cancelled or (state.quiesced and state.peers <= state.notified)
        ):
            self._rooms.pop(room, None)

    def _settle(self, room):
        with self._lock:
            state = self._rooms.get(room)
            if state is None or state.active or state.finalizing:
                return
            if not state.cancelled:
                self._discard(room, state)
                return
            needs_fence = not state.quiesced
            state.finalizing = True
        # Exceptions deliberately propagate to the caller's fail-stop path.
        # Sending an ACK after an uncertain fence would permit unsafe reuse.
        if needs_fence:
            self._fence()
        while True:
            with self._lock:
                state.quiesced = True
                pending = state.peers - state.notified
                failed = state.failed
                if not pending:
                    state.finalizing = False
                    self._discard(room, state)
                    return
            for peer in pending:
                self._notify(room, peer, failed)
            with self._lock:
                state.notified.update(pending)


class DestinationQuiescence:
    """Destination ACK accounting; only the scheduler thread calls this."""

    def __init__(self, expected_sources):
        self.expected = frozenset(expected_sources)
        self.acknowledged = set()
        self.pending = False
        self.started_at = None

    def abort(self, now):
        if not self.pending:
            self.pending = True
            self.started_at = now

    def acknowledge(self, source):
        if source not in self.expected:
            raise RuntimeError("Unexpected Q35rt quiescence source rank")
        self.acknowledged.add(source)

    def ready(self):
        return self.pending and self.expected <= self.acknowledged
