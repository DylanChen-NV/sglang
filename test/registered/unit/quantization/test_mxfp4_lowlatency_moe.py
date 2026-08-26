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

    def test_accepts_kernel_compatible_shapes_and_expert_counts(self):
        for num_experts, hidden_size, intermediate_size in (
            (256, 4096, 512),
            (128, 3072, 768),
            (64, 2048, 2048),
        ):
            method = self._method()
            method.create_weights(
                object(),
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size_per_partition=intermediate_size,
                params_dtype=None,
            )
            self.assertEqual(len(method._fp8.calls), 1)

    def test_rejects_nonpositive_or_unaligned_shapes(self):
        for num_experts, hidden_size, intermediate_size in (
            (0, 4096, 512),
            (256, 4100, 512),
            (256, 4096, 770),
        ):
            method = self._method()
            with self.subTest(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            ):
                with self.assertRaises(ValueError):
                    method.create_weights(
                        object(),
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        intermediate_size_per_partition=intermediate_size,
                        params_dtype=None,
                    )


if __name__ == "__main__":
    unittest.main()
