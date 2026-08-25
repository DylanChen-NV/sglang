#!/bin/bash
set -euo pipefail

project=/home/scratch.ziqingc_gpu/06_codex_projects/09_RS_K2.6
work=/tmp/dsv4_ll_a2a
image=${work}/sglang-nightly-dev-cu13-20260815-a5ba081f.sqsh
result=${project}/artifacts/dsv4_h20_a2a_rqcf0s0a0_20260825.json
correctness=${project}/artifacts/dsv4_h20_activation_a0_correctness_20260825.json

mkdir -p "${work}/home-a0" "${work}/xdg-a0" "${work}/results"
enroot start -r \
    -m "${work}:/work" \
    -m "${project}:/project" \
    -m "${project}/dev/dsv4_h20_lowlatency/sglang:/sgl-workspace/sglang" \
    -m "${project}/dev/dsv4_h20_lowlatency/LowLatencyGroupedGEMM:/work/LowLatencyGroupedGEMM" \
    -m /home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash \
    "${image}" \
    bash -lc 'set -euo pipefail; cd /work/LowLatencyGroupedGEMM; CCACHE_DISABLE=1 MAX_JOBS=1 TORCH_CUDA_ARCH_LIST=9.0a python3 setup.py build_ext --inplace; export HOME=/work/home-a0; export XDG_CACHE_HOME=/work/xdg-a0; export FLASHINFER_WORKSPACE_DIR=/work/cache/flashinfer; export SGLANG_CACHE_DIR=/work/cache/sglang-a0; export PYTHONPATH=/sgl-workspace/sglang/python:/work/LowLatencyGroupedGEMM:${PYTHONPATH:-}; export PYTHONUNBUFFERED=1; unset LOW_LATENCY_MXFP4_PREBUILT; unset ISOLATED_GEMM_SWEEP; python3 /sgl-workspace/sglang/benchmark/dsv4_h20_test_activation_a0.py; python3 /sgl-workspace/sglang/benchmark/dsv4_h20_a2a_a0_variant.py'

cp "${work}/results/a0_correctness.json" "${correctness}"
cp "${work}/results/a2a_results.json" "${result}"
sha256sum "${correctness}" "${result}"
