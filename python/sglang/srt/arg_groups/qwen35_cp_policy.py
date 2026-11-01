"""Startup-only precision policy for the replicated Qwen3.5 CP executor."""

import os

FIRST_TOKEN_POLICY_ENV = "SGLANG_QWEN35_CP_MATCH_FIRST_TOKEN"


def apply_first_token_policy(args):
    """Select a validated arithmetic policy before server argument resolution.

    An explicit environment value owns all four affine settings. Unset keeps
    existing CLI/default behavior. This does not run a native TP oracle or
    promise mathematical argmax equivalence for arbitrary future inputs.
    """
    value = os.environ.get(FIRST_TOKEN_POLICY_ENV)
    if value is None:
        return None
    if value not in ("0", "1"):
        raise ValueError(f"{FIRST_TOKEN_POLICY_ENV} must be exactly 0 or 1")
    if not (args.enable_qwen35_cp_prefill and args.enable_qwen35_cp_prefix_cache):
        raise ValueError(
            f"{FIRST_TOKEN_POLICY_ENV} requires --enable-qwen35-cp-prefill "
            "and --enable-qwen35-cp-prefix-cache"
        )
    if value == "1" and args.enable_tf32_matmul:
        raise ValueError(f"{FIRST_TOKEN_POLICY_ENV}=1 is incompatible with TF32")
    args.qwen35_cp_affine_mode = "fp32" if value == "1" else "two-bf16"
    args.qwen35_cp_fp32_optimization = "u-output" if value == "1" else "none"
    args.qwen35_cp_fp32_refinement = "u16-fused-gemm" if value == "1" else "none"
    args.qwen35_cp_affine_fuse_output = False
    return value
