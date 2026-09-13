"""CPU contract tests for FlashInfer MXFP4 + DeepEP normal dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch

from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
    FlashInferCutlassMxfp4MoeQuantInfo,
    fused_experts_deepep_to_flashinfer_mxfp4,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashInferMxfp4DeepEPNormal(CustomTestCase):
    @staticmethod
    def _quant_info(ep_size=4, ep_rank=3):
        tensor = torch.empty(0)
        return FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=tensor,
            w2_weight=tensor,
            w13_weight_scale=tensor,
            w2_weight_scale=tensor,
            moe_ep_size=ep_size,
            moe_ep_rank=ep_rank,
        )

    def _dispatch_output(self, dtype=torch.bfloat16, with_scale=False):
        return DeepEPNormalDispatchOutput(
            hidden_states=torch.randn(3, 16).to(dtype),
            hidden_states_scale=torch.ones(3, 1) if with_scale else None,
            topk_ids=torch.tensor([[0, -1], [1, 0], [-1, 1]], dtype=torch.int64),
            topk_weights=torch.rand(3, 2),
            num_recv_tokens_per_expert=[2, 2],
        )

    def test_preserves_deepep_routing_contract(self):
        dispatch_output = self._dispatch_output()
        kernel_output = torch.randn(3, 16, dtype=torch.bfloat16)

        with patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
            "fused_experts_none_to_flashinfer_mxfp4",
            return_value=StandardCombineInput(hidden_states=kernel_output),
        ) as run_flashinfer:
            result = fused_experts_deepep_to_flashinfer_mxfp4(
                dispatch_output, self._quant_info(), SimpleNamespace()
            )

        standard_output = run_flashinfer.call_args.args[0]
        local_quant_info = run_flashinfer.call_args.args[1]
        self.assertIs(standard_output.hidden_states, dispatch_output.hidden_states)
        self.assertIs(standard_output.topk_output.topk_ids, dispatch_output.topk_ids)
        self.assertIs(
            standard_output.topk_output.topk_weights, dispatch_output.topk_weights
        )
        self.assertEqual(local_quant_info.moe_ep_size, 1)
        self.assertEqual(local_quant_info.moe_ep_rank, 0)
        self.assertIs(result.hidden_states, kernel_output)
        self.assertIs(result.topk_ids, dispatch_output.topk_ids)
        self.assertIs(result.topk_weights, dispatch_output.topk_weights)

    def test_rejects_fp8_normal_dispatch(self):
        with self.assertRaisesRegex(ValueError, "requires BF16"):
            fused_experts_deepep_to_flashinfer_mxfp4(
                self._dispatch_output(torch.float8_e4m3fn, with_scale=True),
                self._quant_info(),
                SimpleNamespace(),
            )

    def test_empty_rank_skips_flashinfer_kernel(self):
        dispatch_output = DeepEPNormalDispatchOutput(
            hidden_states=torch.empty(0, 16, dtype=torch.bfloat16),
            hidden_states_scale=None,
            topk_ids=torch.empty(0, 2, dtype=torch.int64),
            topk_weights=torch.empty(0, 2),
            num_recv_tokens_per_expert=[0, 0],
        )
        with patch(
            "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass."
            "fused_experts_none_to_flashinfer_mxfp4"
        ) as run_flashinfer:
            result = fused_experts_deepep_to_flashinfer_mxfp4(
                dispatch_output, self._quant_info(), SimpleNamespace()
            )
        run_flashinfer.assert_not_called()
        self.assertEqual(result.hidden_states.shape, (0, 16))

    def test_mxfp4_apply_bypasses_standard_topk_access(self):
        method = object.__new__(Mxfp4MoEMethod)
        method.use_deep_gemm = False
        method._fi_kernel = "cutlass_sm90"
        dispatch_output = self._dispatch_output()
        sentinel = object()

        with patch.object(method, "_apply_sm90_cutlass", return_value=sentinel) as apply:
            result = method.apply(SimpleNamespace(), dispatch_output)

        apply.assert_called_once_with(ANY, dispatch_output)
        self.assertIs(result, sentinel)


if __name__ == "__main__":
    unittest.main()
