"""Regression coverage for Kimi K3 SiTU on FlashInfer SM90 MXFP4."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=1, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_k3_situ_wrapper_builds_per_expert_parameters(monkeypatch):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner import runner as runner_module
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )

    method = Mxfp4FlashinferCutlassMoEMethod.__new__(
        Mxfp4FlashinferCutlassMoEMethod
    )
    method._use_mxfp8_act_scaling = False
    method._mxfp4_weight_global_scale_tensor = None
    method._swiglu_limit_tensor = None
    method._use_swiglu_step = False
    method._situ_beta_tensor = None
    method._situ_linear_beta_tensor = None
    method._use_situ = False

    captured = {}

    def fake_runner(backend, config):
        captured["backend"] = backend
        captured["config"] = config
        return SimpleNamespace()

    monkeypatch.setattr(runner_module, "MoeRunner", fake_runner)
    layer = SimpleNamespace(
        num_local_experts=4,
        w13_weight=torch.empty(1, dtype=torch.uint8, device="cuda"),
    )
    config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=4,
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
    )

    method.create_moe_runner(layer, config)

    assert captured["config"] is config
    assert method._use_situ
    assert not method._use_swiglu_step
    assert method._swiglu_limit_tensor is None
    torch.testing.assert_close(
        method._situ_beta_tensor, torch.full((4,), 4.0, device="cuda")
    )
    torch.testing.assert_close(
        method._situ_linear_beta_tensor, torch.full((4,), 25.0, device="cuda")
    )


def test_k3_base_mxfp4_apply_forwards_situ_parameters():
    """K3 constructs Mxfp4MoEMethod directly; cover that real apply path."""
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    captured = {}

    class CapturingRunner:
        def run(self, dispatch_output, quant_info):
            captured["dispatch_output"] = dispatch_output
            captured["quant_info"] = quant_info
            return SimpleNamespace()

    method = Mxfp4MoEMethod.__new__(Mxfp4MoEMethod)
    method.runner = CapturingRunner()
    method._padded_hidden = 3584
    method.moe_runner_config = MoeRunnerConfig(
        num_experts=4,
        num_local_experts=4,
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
    )
    layer = SimpleNamespace(
        w13_weight=torch.empty(1, dtype=torch.uint8, device="cuda"),
        w2_weight=torch.empty(1, dtype=torch.uint8, device="cuda"),
        w13_weight_scale=torch.empty(1, dtype=torch.uint8, device="cuda"),
        w2_weight_scale=torch.empty(1, dtype=torch.uint8, device="cuda"),
        w13_weight_bias=None,
        w2_weight_bias=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=None,
        situ_beta=torch.full((4,), 4.0, device="cuda"),
        situ_linear_beta=torch.full((4,), 25.0, device="cuda"),
        moe_tp_size=1,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
    )
    dispatch_output = SimpleNamespace()

    method._apply_sm90_cutlass(layer, dispatch_output)

    quant_info = captured["quant_info"]
    assert captured["dispatch_output"] is dispatch_output
    assert quant_info.use_situ
    assert quant_info.situ_beta is layer.situ_beta
    assert quant_info.situ_linear_beta is layer.situ_linear_beta

