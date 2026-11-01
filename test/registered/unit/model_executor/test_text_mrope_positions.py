"""Exercise the real batch method without importing GPU runtime dependencies."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TextMropePositionsTest(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/srt/model_executor/forward_batch_info.py"
        )
        module = ast.parse(source.read_text())
        batch_class = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "ForwardBatch"
        )
        method = next(
            node
            for node in batch_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_compute_mrope_positions"
        )
        # Only its annotations reference the rest of the runtime.
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        cls.execution = SimpleNamespace(
            deterministic=SimpleNamespace(rl_on_policy_target=None)
        )
        namespace = {"torch": torch, "get_exec": lambda: cls.execution}
        isolated = ast.Module(body=[method], type_ignores=[])
        exec(
            compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace
        )
        cls.compute = staticmethod(namespace[method.name])

    def run_batch(self, lengths, prefixes, multimodal=None, decode=False):
        state = SimpleNamespace(
            seq_lens_cpu=torch.tensor([a + b for a, b in zip(lengths, prefixes)]),
            forward_mode=SimpleNamespace(
                is_decode=lambda: decode,
                is_extend=lambda **kwargs: not decode,
            ),
            _expand_mrope_from_input=lambda mm, seq: mm.repeated + seq,
        )
        batch = SimpleNamespace(
            multimodal_inputs=multimodal or [None] * len(lengths),
            extend_lens=lengths,
            prefix_lens=prefixes,
        )
        self.compute(state, SimpleNamespace(device="cpu"), batch)
        self.assertEqual(state.mrope_positions.dtype, torch.int64)
        self.assertTrue(state.mrope_positions.is_contiguous())
        return state.mrope_positions

    def test_text_offsets_empty_and_long_sequences(self):
        for lengths, prefixes in (
            ([11870], [0]),
            ([60193], [0]),
            ([4097], [8192]),
            ([0, 5, 0, 7], [0, 123, 19, 200]),
            ([0], [42]),
        ):
            with self.subTest(lengths=lengths, prefixes=prefixes):
                expected = [
                    p + i for n, p in zip(lengths, prefixes) for i in range(n)
                ]
                actual = self.run_batch(lengths, prefixes)
                self.assertEqual(actual.shape, (3, sum(lengths)))
                self.assertEqual(actual.tolist(), [expected] * 3)

    def test_mixed_multimodal_keeps_distinct_position_axes(self):
        mm = SimpleNamespace(
            mrope_positions=torch.tensor(
                [[100, 101, 102, 103], [200, 201, 202, 203], [300, 301, 302, 303]]
            )
        )
        actual = self.run_batch([2, 2], [7, 1], [None, mm])
        self.assertEqual(
            actual.tolist(),
            [[7, 8, 101, 102], [7, 8, 201, 202], [7, 8, 301, 302]],
        )

    def test_multimodal_empty_slice_fallback(self):
        mm = SimpleNamespace(
            mrope_positions=torch.empty((3, 0), dtype=torch.int64),
            repeated=torch.tensor([[10], [20], [30]]),
        )
        actual = self.run_batch([1], [4], [mm])
        self.assertEqual(actual.tolist(), [[15], [25], [35]])

    def test_rl_target_uses_text_positions(self):
        self.execution.deterministic.rl_on_policy_target = "target"
        try:
            actual = self.run_batch([3], [11], [SimpleNamespace()])
            self.assertEqual(actual.tolist(), [[11, 12, 13]] * 3)
        finally:
            self.execution.deterministic.rl_on_policy_target = None

    def test_decode_positions(self):
        actual = self.run_batch([1, 1], [7, 20], decode=True)
        self.assertEqual(actual.tolist(), [[7, 20]] * 3)


if __name__ == "__main__":
    unittest.main()
