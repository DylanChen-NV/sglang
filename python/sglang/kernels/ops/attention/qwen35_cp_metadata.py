"""Construct per-request CP metadata on the forward stream without H2D waits."""
import torch
import triton
import triton.language as tl


@triton.jit
def _initialize(Data, Initial, length, local_rows, has_initial, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    value = tl.where((i == 0) | (i == 5), length, tl.where(i == 9, local_rows, 0))
    tl.store(Data + i, value, i < 10)
    tl.store(Initial, has_initial)


def create(length, local_rows, has_initial, device):
    # Views own their backing allocation; there is no cross-request cache.
    # FlashInfer validates alignment of each metadata view, not only its base.
    data = torch.empty(10, dtype=torch.int32, device=device)
    initial = torch.empty(1, dtype=torch.bool, device=device)
    _initialize[(1,)](data, initial, length, local_rows, has_initial, BLOCK=32, num_warps=1)
    return data[:1], data[4:6], data[8:10], initial
