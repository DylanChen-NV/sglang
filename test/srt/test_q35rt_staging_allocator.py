"""CPU regressions for variable-sized staging ring wraps and live leases."""

import threading
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.staging_buffer import StagingAllocator
from sglang.srt.disaggregation.common.staging_handler import is_watermark_ready


class TestStagingAllocatorWatermark(unittest.TestCase):
    def allocator(self):
        # Exercise the actual ring logic without allocating a CUDA buffer.
        value = object.__new__(StagingAllocator)
        value.total_size = 100
        value.head = value.round = value.next_alloc_id = 0
        value.watermark_round = value.watermark_tail = 0
        value.allocations, value.alloc_order = {}, []
        value.lock = threading.Lock()
        return value

    def ready(self, allocator, allocation, size):
        _, offset, generation = allocation
        state = SimpleNamespace(remote_watermarks={"peer": allocator.get_watermark()})
        return is_watermark_ready(state, "peer", generation, offset + size)

    def test_empty_ring_accepts_larger_next_prefix(self):
        allocator = self.allocator()
        previous = allocator.get_watermark()
        for size in (70, 60, 75, 90, 45, 99):
            allocation = allocator.assign(size)
            self.assertTrue(self.ready(allocator, allocation, size))
            allocator.free(allocation[0])
            self.assertGreater(allocator.get_watermark(), previous)
            previous = allocator.get_watermark()
            allocator.free(allocation[0])
            self.assertEqual(allocator.get_watermark(), previous)

    def test_wrap_waits_for_live_allocation(self):
        allocator = self.allocator()
        first = allocator.assign(70)
        second = allocator.assign(60)
        self.assertFalse(self.ready(allocator, second, 60))
        allocator.free(first[0])
        self.assertTrue(self.ready(allocator, second, 60))
        third = allocator.assign(50)
        self.assertFalse(self.ready(allocator, third, 50))
        allocator.free(second[0])
        self.assertTrue(self.ready(allocator, third, 50))

    def test_freed_empty_generation_cannot_authorize_new_live_wrap(self):
        allocator = self.allocator()
        initial = allocator.assign(10)
        allocator.free(initial[0])
        first, second = allocator.assign(30), allocator.assign(40)
        wrapped = allocator.assign(50)
        self.assertFalse(self.ready(allocator, wrapped, 50))
        allocator.free(first[0])
        self.assertFalse(self.ready(allocator, wrapped, 50))
        allocator.free(second[0])
        self.assertTrue(self.ready(allocator, wrapped, 50))


if __name__ == "__main__":
    unittest.main()
