"""CPU-only tests of the production config reader and dispatch eligibility.

Extract the actual definitions to avoid importing CUDA-dependent SGLang modules.
No routing logic is duplicated here; GPU validation imports the full backend.
"""

import ast
import logging
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SOURCE = (
    Path(__file__).resolve().parents[5]
    / "python/sglang/srt/layers/attention/nvfp4_sm120_backend.py"
)
tree = ast.parse(SOURCE.read_text())
definitions = ast.parse("from __future__ import annotations").body
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_read_nvfp4_min_seq_len":
        definitions.append(node)
    if isinstance(node, ast.ClassDef) and node.name == "NVFP4SM120AttnBackend":
        definitions.extend(
            method
            for method in node.body
            if isinstance(method, ast.FunctionDef)
            and method.name == "_unsupported_reason"
        )

logger = logging.getLogger("nvfp4-routing-test")
cuda = SimpleNamespace(is_current_stream_capturing=lambda: False)
namespace = {
    "os": os,
    "logger": logger,
    "torch": SimpleNamespace(cuda=cuda, float16="float16", bfloat16="bfloat16"),
    "ForwardMode": SimpleNamespace(EXTEND="extend"),
    "AttentionType": SimpleNamespace(DECODER="decoder"),
}
exec(
    compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
read_min_seq_len = namespace["_read_nvfp4_min_seq_len"]
unsupported_reason = namespace["_unsupported_reason"]


class TestNVFP4MinimumConfig(unittest.TestCase):
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(read_min_seq_len(), 16384)

    def test_override_and_zero(self):
        for value in ("0", "4096", "32768"):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SGLANG_NVFP4_ATTN_MIN_SEQ_LEN": value}, clear=True
            ):
                self.assertEqual(read_min_seq_len(), int(value))

    def test_invalid_values(self):
        for value in ("-1", "text", "1.5", ""):
            with self.subTest(value=value), patch.dict(
                os.environ, {"SGLANG_NVFP4_ATTN_MIN_SEQ_LEN": value}, clear=True
            ):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    read_min_seq_len()

    def test_legacy_max_warns_and_is_not_a_minimum(self):
        with patch.dict(
            os.environ, {"SGLANG_NVFP4_ATTN_MAX_SEQ_LEN": "8192"}, clear=True
        ), self.assertLogs(logger, level="WARNING") as captured:
            self.assertEqual(read_min_seq_len(), 16384)
        self.assertIn("deprecated and ignored", captured.output[0])

    def test_explicit_min_wins_over_legacy_max(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_NVFP4_ATTN_MAX_SEQ_LEN": "8192",
                "SGLANG_NVFP4_ATTN_MIN_SEQ_LEN": "32768",
            },
            clear=True,
        ), self.assertLogs(logger, level="WARNING"):
            self.assertEqual(read_min_seq_len(), 32768)


class TestNVFP4MinimumRouting(unittest.TestCase):
    def setUp(self):
        self.backend = SimpleNamespace(
            _nvfp4_is_draft_worker=False,
            _nvfp4_context_parallel_size=1,
            _nvfp4_min_seq_len=16384,
        )
        self.layer = SimpleNamespace(
            is_cross_attention=False,
            attn_type="decoder",
            sliding_window_size=-1,
            logit_cap=0,
            qk_head_dim=256,
            v_head_dim=256,
            tp_q_head_num=8,
            tp_k_head_num=1,
            tp_v_head_num=1,
        )

    def reason(self, seq_lens, *, prefix_lens=None, token_count=None, kwargs=None):
        tokens = sum(seq_lens) if token_count is None else token_count
        q = SimpleNamespace(ndim=2, shape=(tokens, 2048), dtype="bfloat16")
        k = SimpleNamespace(ndim=3, shape=(tokens, 1, 256), dtype="bfloat16")
        v = SimpleNamespace(ndim=3, shape=(tokens, 1, 256), dtype="bfloat16")
        batch = SimpleNamespace(
            forward_mode="extend",
            spec_info=None,
            mha_return_lse=False,
            extend_prefix_lens_cpu=prefix_lens or [0] * len(seq_lens),
            extend_seq_lens_cpu=seq_lens,
            out_cache_loc=object(),
        )
        return unsupported_reason(
            self.backend, q, k, v, self.layer, batch, kwargs or {}, True
        )

    def test_short_sequences_fall_back(self):
        for length in (1, 4096, 8192, 16383):
            with self.subTest(length=length):
                self.assertEqual(self.reason([length]), "sequence_too_short")

    def test_threshold_is_inclusive_and_no_maximum(self):
        for length in (16384, 16385, 32768, 65536):
            with self.subTest(length=length):
                self.assertIsNone(self.reason([length]))

    def test_mixed_batch_falls_back_in_either_order(self):
        for seq_lens in ([8192, 32768], [32768, 8192]):
            self.assertEqual(self.reason(seq_lens), "sequence_too_short")

    def test_all_long_batch_remains_eligible(self):
        self.assertIsNone(self.reason([16384, 32768]))

    def test_zero_minimum_allows_short_sequences(self):
        self.backend._nvfp4_min_seq_len = 0
        self.assertIsNone(self.reason([1]))
        self.assertEqual(self.reason([0]), "empty_sequence")

    def test_cached_prefix_still_falls_back(self):
        self.assertEqual(self.reason([32768], prefix_lens=[1]), "cached_prefix")

    def test_invalid_token_count_still_falls_back(self):
        self.assertEqual(
            self.reason([32768], token_count=32767), "token_count_mismatch"
        )

    def test_other_guards_are_preserved(self):
        self.backend._nvfp4_is_draft_worker = True
        self.assertEqual(self.reason([32768]), "draft_worker")
        self.backend._nvfp4_is_draft_worker = False
        self.backend._nvfp4_context_parallel_size = 2
        self.assertEqual(self.reason([32768]), "context_parallel")
        self.backend._nvfp4_context_parallel_size = 1
        self.layer.qk_head_dim = 128
        self.assertEqual(self.reason([32768]), "unsupported_head_shape")
        self.layer.qk_head_dim = 256
        self.assertEqual(
            self.reason([32768], kwargs={"extra": True}), "extra_attention_kwargs"
        )

    def test_graph_capture_still_falls_back(self):
        with patch.object(cuda, "is_current_stream_capturing", return_value=True):
            self.assertEqual(self.reason([32768]), "cuda_graph_capture")


if __name__ == "__main__":
    unittest.main()
