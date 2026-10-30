"""Construct per-request CP metadata on the forward stream without H2D waits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _initialize(Data, Initial, length, local_rows, has_initial, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.where((i == 0) | (i == 2), length, tl.where(i == 4, local_rows, 0))
    tl.store(Data + i, value, i < 5)
    tl.store(Initial, has_initial)


def create(length, local_rows, has_initial, device):
    # Views own their backing allocation; there is no cross-request cache.
    data = torch.empty(5, dtype=torch.int32, device=device)
    initial = torch.empty(1, dtype=torch.bool, device=device)
    _initialize[(1,)](data, initial, length, local_rows, has_initial, BLOCK=32, num_warps=1)
    return data[:1], data[1:3], data[3:5], initial
