import unittest

from sglang.srt.layers.quantization.mxfp4_lowlatency_moe import (
    Mxfp4LowLatencyMoEMethod,
)


class _FakeFp8Method:
    def __init__(self):
        self.calls = []

    def create_weights(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestMxfp4LowLatencyMoEShapes(unittest.TestCase):
    def _method(self):
        method = object.__new__(Mxfp4LowLatencyMoEMethod)
        method._fp8 = _FakeFp8Method()
        return method

    def test_accepts_dsv4_tp2_and_tp4_shapes(self):
        for intermediate_size in (512, 1024):
            method = self._method()
            method.create_weights(
                object(),
                num_experts=256,
                hidden_size=4096,
                intermediate_size_per_partition=intermediate_size,
                params_dtype=None,
            )
            self.assertEqual(len(method._fp8.calls), 1)

    def test_rejects_other_intermediate_sizes(self):
        method = self._method()
        with self.assertRaisesRegex(ValueError, "TP2/TP4"):
            method.create_weights(
                object(),
                num_experts=256,
                hidden_size=4096,
                intermediate_size_per_partition=2048,
                params_dtype=None,
            )


if __name__ == "__main__":
    unittest.main()
