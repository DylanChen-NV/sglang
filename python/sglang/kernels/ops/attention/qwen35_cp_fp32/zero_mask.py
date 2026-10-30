"""Skip homogeneous CP correction tiles that contain only signed BF16 zero.

The current executor has one active request. Buffers are per device, all work runs
on the caller's stream, and no cache/state tensor is changed by this preparation.
"""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def _finite_state(State, Valid, BLOCK: tl.constexpr = 16384):
    head = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    values = tl.load(State + head * 16384 + index)
    bits = values.to(tl.int32, bitcast=True) & 0x7FFFFFFF
    invalid = tl.sum((bits >= 0x7F800000).to(tl.int32), 0)
    tl.store(Valid + head, (invalid == 0).to(tl.int32))


@triton.jit
def _zero_tiles(U, Out, Valid, Flags, E: tl.constexpr, TILES: tl.constexpr):
    tile, head = tl.program_id(0), tl.program_id(1)
    rows = tile * 64 + tl.arange(0, 64)
    keys = tl.arange(0, 128)
    offset = (rows[:, None] * 32 + head) * 128 + keys[None, :]
    values = tl.load(U + offset, rows[:, None] < E, other=0)
    bits = values.to(tl.uint16, bitcast=True) & 0x7FFF
    nonzero = tl.sum(tl.sum((bits != 0).to(tl.int32), 1), 0)
    valid = tl.load(Valid + head)
    active = tl.full((), 1, tl.int32)
    if (nonzero == 0) & (valid != 0):
        old = tl.load(Out + offset, rows[:, None] < E, other=0)
        old_bits = old.to(tl.uint16, bitcast=True)
        nan_count = tl.sum(tl.sum(((old_bits & 0x7FFF) > 0x7F80).to(tl.int32), 1), 0)
        if nan_count == 0:
            # The original +0 accumulator and final addition turn -0 into +0.
            tl.store(Out + offset, 0, (old_bits == 0x8000) & (rows[:, None] < E))
            active = tl.full((), 0, tl.int32)
    tl.store(Flags + head * TILES + tile, active)


class _Workspace:
    def __init__(self, device):
        self.device = device
        self.capacity = 0
        self.flags = None
        self.valid = torch.empty(32, dtype=torch.int32, device=device)

    def prepare(self, out, u, state):
        tiles = triton.cdiv(len(u), 64)
        if tiles > self.capacity:
            self.flags = torch.empty(32 * tiles, dtype=torch.int32, device=self.device)
            self.capacity = tiles
        _finite_state[(32,)](state, self.valid, num_warps=4)
        _zero_tiles[(tiles, 32)](u, out, self.valid, self.flags, len(u), tiles, num_warps=8)
        return self.flags


@functools.lru_cache(maxsize=4)
def _workspace(device):
    return _Workspace(device)


def prepare(out, u, state):
    return _workspace(out.device).prepare(out, u, state)
