"""GPU layout/graph tests; no LowLatencyGroupedGEMM is imported."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMxfp4MoeQuantInfo,
    fused_experts_deepep_to_flashinfer_mxfp4,
)
from sglang.srt.layers.moe.moe_runner.flashinfer_humming_deepep_ll import (
    prepare_ll_rows,
    run_deepep_ll_humming,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.token_dispatcher import deepep as dispatcher


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlashInferDeepEPLL(unittest.TestCase):
    def setUp(self):
        self.experts, self.capacity, self.hidden = 4, 5, 128
        self.x = torch.randn(4, 5, 128, device="cuda", dtype=torch.bfloat16)
        self.counts = torch.tensor([2, 0, 5, 1], device="cuda", dtype=torch.int32)
        self.ids = torch.tensor([[0, 7]], device="cuda", dtype=torch.int64)
        self.weights = torch.tensor([[0.25, 0.75]], device="cuda")
        empty = torch.empty(4, 0, device="cuda")
        scale = torch.ones(4, device="cuda")
        self.qi = FlashInferCutlassMxfp4MoeQuantInfo(
            empty, empty, empty, empty,
            w13_humming_residual_scale=scale,
            w2_humming_residual_scale=scale,
            humming_fc2_act_scale=torch.ones((), device="cuda"),
            moe_ep_size=2, moe_ep_rank=1,
        )
        self.config = MoeRunnerConfig(top_k=2, activation="situ")

    def dispatch(self, x=None, sf=None):
        return DeepEPLLDispatchOutput(
            self.x if x is None else x, sf, self.ids, self.weights,
            self.counts, 2,
        )

    def valid_mask(self):
        return torch.arange(5, device="cuda")[None, :] < self.counts[:, None]

    def test_padding_poison_and_local_namespace(self):
        self.x[~self.valid_mask()] = float("nan")
        prepared = prepare_ll_rows(self.dispatch(), self.qi)
        expected_ids = torch.arange(4, device="cuda")[:, None].expand(4, 5).clone()
        expected_ids[~self.valid_mask()] = -1
        torch.testing.assert_close(prepared.topk_output.topk_ids[:, 0], expected_ids.flatten().int())
        y = prepared.hidden_states.view_as(self.x)
        torch.testing.assert_close(y[self.valid_mask()], self.x[self.valid_mask()])
        self.assertTrue(torch.isfinite(y).all().item())
        self.assertEqual(y[~self.valid_mask()].abs().sum().item(), 0)
        self.assertEqual(prepared.topk_output.topk_weights.sum().item(), 8)

    def test_strided_repeated_fp8_scales(self):
        # Carrier includes two groups, and the physical token scale stride is
        # non-contiguous. This emulates DeepEP's group-major scale allocation.
        x = self.x.to(torch.float8_e4m3fn)
        sf = torch.rand(4, 2, 5, device="cuda").transpose(1, 2)
        sf[..., 1] = sf[..., 0]
        x = torch.cat([x.float(), x.float()], -1).to(torch.float8_e4m3fn)
        prepared = prepare_ll_rows(self.dispatch(x, sf), self.qi)
        # Communication FP8 bytes must not be requantized by the adapter.
        actual_bytes = prepared.hidden_states.view_as(x).view(torch.uint8)
        torch.testing.assert_close(actual_bytes[self.valid_mask()], x.view(torch.uint8)[self.valid_mask()])
        self.assertEqual(actual_bytes[~self.valid_mask()].sum().item(), 0)
        actual = prepared.hidden_states_scale.view(4, 5)
        torch.testing.assert_close(actual[self.valid_mask()], sf[..., 0][self.valid_mask()])
        torch.testing.assert_close(actual[~self.valid_mask()], torch.ones_like(actual[~self.valid_mask()]))

    def test_no_weight_double_application_and_padded_output(self):
        calls = []

        def mock_humming(standard, qi, cfg):
            calls.append((qi.moe_ep_size, qi.moe_ep_rank))
            return StandardCombineInput(standard.hidden_states * 2)

        result = run_deepep_ll_humming(self.dispatch(), self.qi, self.config, mock_humming)
        self.assertEqual(calls, [(1, 0)])
        self.assertIs(result.topk_ids, self.ids)
        self.assertIs(result.topk_weights, self.weights)
        torch.testing.assert_close(result.hidden_states[self.valid_mask()], self.x[self.valid_mask()] * 2)
        self.assertEqual(result.hidden_states[~self.valid_mask()].abs().sum().item(), 0)

    def test_graph_replay_reads_updated_device_counts(self):
        def mock_humming(standard, qi, cfg):
            return StandardCombineInput(standard.hidden_states.clone())

        for _ in range(3):
            run_deepep_ll_humming(self.dispatch(), self.qi, self.config, mock_humming)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run_deepep_ll_humming(self.dispatch(), self.qi, self.config, mock_humming)
        self.counts.copy_(torch.tensor([0, 3, 1, 5], device="cuda", dtype=torch.int32))
        self.x.fill_(3)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(result.hidden_states[self.valid_mask()], self.x[self.valid_mask()])
        self.assertEqual(result.hidden_states[~self.valid_mask()].abs().sum().item(), 0)

    def test_all_experts_empty(self):
        self.counts.zero_()
        result = prepare_ll_rows(self.dispatch(), self.qi)
        self.assertEqual(result.hidden_states.abs().sum().item(), 0)
        self.assertTrue((result.topk_output.topk_ids == -1).all().item())

    def test_empty_physical_carrier_skips_gemm(self):
        self.counts.zero_()
        dispatch = self.dispatch(torch.empty(4, 0, 128, device="cuda", dtype=torch.bfloat16))
        result = run_deepep_ll_humming(
            dispatch, self.qi, self.config,
            lambda *args: self.fail("empty physical carrier must skip GEMM"),
        )
        self.assertEqual(result.hidden_states.shape, (4, 0, 128))

    def test_w4a16_rejected(self):
        self.qi.humming_fc2_act_scale = None
        with self.assertRaisesRegex(ValueError, "Humming W4A8"):
            fused_experts_deepep_to_flashinfer_mxfp4(self.dispatch(), self.qi, self.config)

    def test_dispatcher_humming_fp8_uses_per_token_transport(self):
        calls = []

        class FakeBuffer:
            def low_latency_dispatch(self, x, ids, capacity, experts,
                                     use_per_token_scale=False, **kwargs):
                calls.append((x, use_per_token_scale, kwargs))
                return self_recv, self_counts, None, None, None

        self_recv, self_counts = self.x, self.counts
        impl = object.__new__(dispatcher._DeepEPDispatcherImplLowLatency)
        impl.quant_config = {}
        impl.use_fp8, impl.use_nvfp4 = True, False
        impl.low_latency_quant_mode = None
        impl._per_token_scale_runtime_checked = False
        impl.num_max_dispatch_tokens_per_rank = 5
        impl.num_experts = 4
        impl.return_recv_hook = True
        backend = SimpleNamespace(is_lowlatency_mxfp4=lambda: False,
                                  is_flashinfer_mxfp4=lambda: True)
        quant_calls = []

        def fake_quant(x, q, sf):
            quant_calls.append(x)
            q.copy_(x.to(torch.float8_e4m3fn))
            sf.fill_(1)

        with patch.object(impl, "_get_buffer", return_value=FakeBuffer()), \
             patch.object(dispatcher, "get_moe_runner_backend", return_value=backend), \
             patch.object(dispatcher, "is_blackwell", return_value=False), \
             patch.object(dispatcher, "_deepep_precompile_tp_barrier"), \
             patch("sglang.kernels.ops.quantization.sgl_per_token_quant_fp8", side_effect=fake_quant):
            impl._dispatch_core(self.x.flatten(0, 1), self.ids, self.weights)
        self.assertEqual(len(quant_calls), 1)
        self.assertTrue(calls[0][1])
        self.assertTrue(calls[0][2]["use_fp8"])
        self.assertEqual(calls[0][0][0].dtype, torch.float8_e4m3fn)
        self.assertEqual(calls[0][0][1].shape, (20, 1))
        self.assertTrue(impl._per_token_scale_runtime_checked)


if __name__ == "__main__":
    unittest.main()
