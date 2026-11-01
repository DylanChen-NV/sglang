"""Per-device scratch for the single-stream, single-active-request executor."""

import functools

import torch


class _Workspace:
    def __init__(self, device):
        self.device = device
        self.capacity = 0
        self.flags = None

    def flags_for(self, rows):
        count = 32 * ((rows + 63) // 64 + 1)
        if count > self.capacity:
            self.flags = torch.empty(count, dtype=torch.int32, device=self.device)
            self.capacity = count
        return self.flags


@functools.lru_cache(maxsize=4)
def _workspace(device):
    return _Workspace(device)


def flags_for(device, rows):
    return _workspace(device).flags_for(rows)
