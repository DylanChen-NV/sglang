"""CPU regressions for W4AFP8 DeepEP dispatcher dtypes."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    DispatcherOutputDtype,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization import w4afp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakePerTokenDeepEPBuffer:
    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        use_per_token_scale=False,
        async_finish=False,
        return_recv_hook=False,
        **kwargs,
    ):
        self.hidden_states = hidden_states
        self.topk_ids = topk_ids
        self.kwargs = dict(
            use_fp8=use_fp8,
            use_per_token_scale=use_per_token_scale,
            async_finish=async_finish,
            return_recv_hook=return_recv_hook,
            **kwargs,
        )
        return hidden_states, torch.zeros(2, dtype=torch.int32), object(), object(), Mock()


class _FakeLegacyDeepEPBuffer:
    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        async_finish=False,
        return_recv_hook=False,
    ):
        raise AssertionError("legacy DeepEP dispatch should be rejected before launch")


class TestW4AFP8DeepEPDispatcherDtype(CustomTestCase):
    def test_w4afp8_sets_mode_specific_dispatcher_dtypes(self):
        dispatcher = Mock()
        layer = SimpleNamespace(
            dispatcher=dispatcher,
            w2_weight=torch.empty(0),
            w13_weight_scale_inv=torch.ones((1, 1, 4)),
            w2_weight_scale_inv=torch.ones((1, 1, 4)),
            w13_input_scale=torch.ones(1),
            w2_input_scale=torch.ones(1),
        )

        w4afp8.W4AFp8MoEMethod(SimpleNamespace()).process_weights_after_loading(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "fp8",
            }
        )

    def test_mode_specific_dtype_selection(self):
        quant_config = {
            "normal_dispatcher_output_dtype": "bf16",
            "low_latency_dispatcher_output_dtype": "fp8",
        }

        with (
            patch.object(moe_utils, "get_server_args", return_value=None),
            patch.object(
                moe_utils.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                "get",
                return_value=False,
            ),
            patch.object(
                moe_utils,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
        ):
            normal_dtype = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(
                    quant_config=quant_config,
                    dispatch_mode=DeepEPMode.NORMAL,
                )
            )
            low_latency_dtype = moe_utils.get_deepep_output_dtype(
                SimpleNamespace(
                    quant_config=quant_config,
                    dispatch_mode=DeepEPMode.LOW_LATENCY,
                )
            )

        self.assertEqual(
            deepep._DeepEPDispatcherImplNormal.dispatch_mode, DeepEPMode.NORMAL
        )
        self.assertEqual(
            deepep._DeepEPDispatcherImplLowLatency.dispatch_mode,
            DeepEPMode.LOW_LATENCY,
        )
        self.assertEqual(normal_dtype, DispatcherOutputDtype.BF16)
        self.assertEqual(low_latency_dtype, DispatcherOutputDtype.FP8)

    def test_normal_rejects_fp8_and_preserves_empty_bf16(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        empty_topk_ids = torch.empty((0, 1), dtype=torch.int64)
        empty_topk_weights = torch.empty((0, 1), dtype=torch.float32)

        fp8_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.float8_e4m3fn),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        with self.assertRaisesRegex(RuntimeError, "requires BF16"):
            method.apply_deepep_normal(SimpleNamespace(), fp8_dispatch_output)

        bf16_dispatch_output = SimpleNamespace(
            hidden_states=torch.empty((0, 128), dtype=torch.bfloat16),
            topk_ids=empty_topk_ids,
            topk_weights=empty_topk_weights,
        )
        output = method.apply_deepep_normal(SimpleNamespace(), bf16_dispatch_output)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(output.shape, (0, 128))

    @staticmethod
    def _per_token_dispatcher(buffer):
        impl = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        impl.quant_config = {}
        impl.low_latency_quant_mode = None
        impl._low_latency_quant_mode_runtime_checked = False
        impl.use_fp8 = True
        impl.use_nvfp4 = False
        impl._per_token_scale_runtime_checked = False
        impl.return_recv_hook = False
        impl.num_max_dispatch_tokens_per_rank = 2
        impl.num_experts = 2
        impl._get_buffer = Mock(return_value=buffer)
        return impl

    def test_lowlatency_mxfp4_reuses_b1_per_token_quant_before_dispatch(self):
        from sglang.kernels.ops import quantization

        buffer = _FakePerTokenDeepEPBuffer()
        impl = self._per_token_dispatcher(buffer)
        hidden_states = torch.randn((2, 128), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)

        with (
            patch.object(
                deepep,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.LOWLATENCY_MXFP4,
            ),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(quantization, "sgl_per_token_quant_fp8") as quant,
        ):
            impl._dispatch_core(hidden_states, topk_ids, topk_weights)

        quant.assert_called_once()
        quant_input, quant_output, quant_scale = quant.call_args.args
        self.assertIs(quant_input, hidden_states)
        self.assertIs(buffer.hidden_states[0], quant_output)
        self.assertIs(buffer.hidden_states[1], quant_scale)
        self.assertEqual(quant_output.dtype, torch.float8_e4m3fn)
        self.assertEqual(quant_scale.dtype, torch.float32)
        self.assertEqual(tuple(quant_scale.shape), (2, 1))
        self.assertTrue(buffer.kwargs["use_fp8"])
        self.assertTrue(buffer.kwargs["use_per_token_scale"])

    def test_lowlatency_mxfp4_rejects_legacy_deepep_runtime(self):
        impl = self._per_token_dispatcher(_FakeLegacyDeepEPBuffer())
        hidden_states = torch.randn((2, 128), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0], [1]], dtype=torch.int64)
        topk_weights = torch.ones((2, 1), dtype=torch.float32)

        with (
            patch.object(
                deepep,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.LOWLATENCY_MXFP4,
            ),
            self.assertRaisesRegex(
                RuntimeError, "per-token-scale DeepEP runtime"
            ),
        ):
            impl._dispatch_core(hidden_states, topk_ids, topk_weights)

    def test_low_latency_requires_fp8_scales(self):
        method = w4afp8.W4AFp8MoEMethod(SimpleNamespace())
        dispatch_output = (
            torch.empty((1, 1, 128), dtype=torch.bfloat16),
            None,
            torch.empty((0, 1), dtype=torch.int64),
            torch.empty((0, 1), dtype=torch.float32),
            torch.zeros(1, dtype=torch.int32),
            0,
        )

        with self.assertRaisesRegex(RuntimeError, "requires FP8"):
            method.apply_deepep_ll(SimpleNamespace(), dispatch_output)


if __name__ == "__main__":
    unittest.main()
