#!/usr/bin/env bash
set -euo pipefail

mode="${1:-compare}"
if [[ "$mode" == compare ]]; then
  compare_rc=0
  for compare_mode in b1 f1; do
    if ! K3_RUN_BENCH="${K3_RUN_BENCH:-1}" "$0" "$compare_mode"; then
      compare_rc=1
    fi
  done
  exit "$compare_rc"
fi
case "$mode" in
  b1|f1) ;;
  *) echo "usage: $0 {b1|f1|compare}" >&2; exit 2 ;;
esac

base=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/kimi-k3-deepep-lowlatency/b1-validation
model=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/models/Kimi-K3-official-f831ab6
run_id="${K3_RUN_ID:-$SLURM_JOB_ID}"
run_dir="$base/results/full-k3-b1-f1-${run_id}-${mode}"
mem_fraction_static="${K3_MEM_FRACTION_STATIC:-0.84}"
max_running_requests="${K3_MAX_RUNNING_REQUESTS:-128}"
max_mamba_cache_size="${K3_MAX_MAMBA_CACHE_SIZE:-640}"
cuda_graph_bs="${K3_CUDA_GRAPH_BS:-128}"
bench_num_prompts="${K3_BENCH_NUM_PROMPTS:-128}"
gsp_prompts_per_group="${K3_GSP_PROMPTS_PER_GROUP:-128}"
gsp_system_prompt_len="${K3_GSP_SYSTEM_PROMPT_LEN:-99999}"
gsp_question_len="${K3_GSP_QUESTION_LEN:-1}"
gsp_output_len="${K3_GSP_OUTPUT_LEN:-128}"
max_concurrency="${K3_MAX_CONCURRENCY:-128}"
capacity_gate_min="${K3_CAPACITY_GATE_MIN:-100006}"
expected_input_tokens="$((bench_num_prompts * (gsp_system_prompt_len + gsp_question_len)))"
case "$mode" in
  b1)
    port=30111
    deepep_dtype=bf16
    ;;
  f1)
    port=30112
    deepep_dtype=fp8
    ;;
esac
server_extra_args=()
bench_extra_args=()
if [[ "${K3_SKIP_SERVER_WARMUP:-0}" == 1 ]]; then
  server_extra_args+=(--skip-server-warmup)
fi
if [[ "${K3_FAKE_PREFILL:-0}" == 1 ]]; then
  server_extra_args+=(--disaggregation-mode decode --disaggregation-transfer-backend fake)
  bench_extra_args+=(--fake-prefill)
fi
local_rank="${SLURM_NODEID}"
rank="$((local_rank + ${K3_NODE_RANK_OFFSET:-0}))"
mkdir -p "$run_dir"

sglang_src="${K3_SGLANG_SRC:-$base/sglang}"
llgg_src="${K3_LLGG_SRC:-$base/LowLatencyGroupedGEMM}"
llgg_build="${K3_LLGG_BUILD:-$base/build/lowlatency-extension-c40c108}"
export PYTHONPATH="$sglang_src/python:$llgg_build:${PYTHONPATH:-}"
export SGLANG_LOWLATENCY_DEEPEP_LAYOUT=compact
export SGLANG_LOWLATENCY_MXFP4_VARIANT=final
export SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS="${SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS:-528}"

# Compilation caches are node-local. Sharing them across 32 ranks on Lustre
# serializes atomic renames and can make the first K3 warmup exceed 10 minutes.
node_cache="/tmp/k3-b1-f1-jit-${SLURM_JOB_ID}-${mode}"
mkdir -p "$node_cache"
export XDG_CACHE_HOME="$node_cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$node_cache/torchinductor"
export TRITON_CACHE_DIR="$node_cache/triton"
export SGLANG_CACHE_DIR="$base/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$node_cache/sglang-jit"
export FLASHINFER_WORKSPACE_DIR="$base/cache/flashinfer"
export SGLANG_WARMUP_TIMEOUT=1800
export TMPDIR=/tmp
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -c 'import torch; assert torch.cuda.device_count() == 8; assert all(torch.cuda.get_device_name(i).startswith("NVIDIA H100") for i in range(8)); [torch.empty(1, device=f"cuda:{i}") for i in range(8)]' >"$run_dir/gpu_probe_rank${rank}.log" 2>&1

if [[ "$rank" == 0 ]]; then
  hostname >"$run_dir/master_host"
  {
    echo "mode=$mode"
    echo "run_id=$run_id"
    echo "allocation_job_id=$SLURM_JOB_ID"
    echo "dispatcher_output_dtype=$deepep_dtype"
    echo "sglang_sha=$(git -C "$sglang_src" rev-parse HEAD)"
    echo "llgg_sha=$(git -C "$llgg_src" rev-parse HEAD)"
    echo "llgg_build=$llgg_build"
  } >"$run_dir/code_manifest.txt"
fi
for _ in $(seq 1 120); do
  [[ -s "$run_dir/master_host" ]] && break
  sleep 1
done
master_host=$(head -1 "$run_dir/master_host")

python3 -m sglang.launch_server \
  "${server_extra_args[@]}" \
  --model-path "$model" \
  --tokenizer-path "$model" \
  --trust-remote-code \
  --random-seed 42 \
  --language-only \
  --tp-size 32 \
  --ep-size 32 \
  --nnodes 4 \
  --node-rank "$rank" \
  --dist-init-addr "${master_host}:20000" \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 131072 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests "$max_running_requests" \
  --max-mamba-cache-size "$max_mamba_cache_size" \
  --mamba-ssm-dtype bfloat16 \
  --enable-shared-experts-attn-tp \
  --mem-fraction-static "$mem_fraction_static" \
  --cuda-graph-backend-decode full \
  --cuda-graph-max-bs-decode "$cuda_graph_bs" \
  --cuda-graph-bs-decode "$cuda_graph_bs" \
  --cuda-graph-backend-prefill disabled \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 \
  --port "$port" \
  --moe-runner-backend lowlatency_mxfp4 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --deepep-dispatcher-output-dtype "$deepep_dtype" \
  >"$run_dir/rank${rank}.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

if [[ "$rank" == 0 ]]; then
  ready=0
  for _ in $(seq 1 1800); do
    if curl -fsS "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if [[ "$ready" == 1 ]]; then
    max_total_tokens=$(grep -a -o 'max_total_num_tokens=[0-9]*' "$run_dir/rank0.log" | tail -1 | cut -d= -f2)
    if [[ -z "$max_total_tokens" || "$max_total_tokens" -lt "$capacity_gate_min" ]]; then
      echo "Insufficient token capacity: max_total_num_tokens=${max_total_tokens:-missing}" >"$run_dir/capacity_gate.log"
      echo FAIL >"$run_dir/status"
      touch "$run_dir/done"
      exit 1
    fi
    if [[ "${K3_FAKE_PREFILL:-0}" != 1 ]]; then
      curl -fsS "http://127.0.0.1:${port}/generate" \
        -H 'Content-Type: application/json' \
        -d '{"text":"Hello","sampling_params":{"temperature":0,"max_new_tokens":2}}' \
        >"$run_dir/response.json"
    fi
    if [[ "${K3_RUN_BENCH:-1}" == 1 ]]; then
      python3 -m sglang.bench_serving \
        --backend sglang \
        "${bench_extra_args[@]}" \
        --host 127.0.0.1 \
        --port "$port" \
        --dataset-name generated-shared-prefix \
        --tokenizer "$model" \
        --model "$model" \
        --num-prompts "$bench_num_prompts" \
        --gsp-num-groups 1 \
        --gsp-prompts-per-group "$gsp_prompts_per_group" \
        --gsp-system-prompt-len "$gsp_system_prompt_len" \
        --gsp-question-len "$gsp_question_len" \
        --gsp-output-len "$gsp_output_len" \
        --gsp-range-ratio 1.0 \
        --gsp-fast-prepare \
        --gsp-ordered \
        --max-concurrency "$max_concurrency" \
        --request-rate inf \
        --warmup-requests 1 \
        --seed 42 \
        --output-details \
        --output-file "$run_dir/bench_shared100k_bs128.jsonl" \
        >"$run_dir/bench_shared100k_bs128.log" 2>&1 || {
          echo FAIL >"$run_dir/status"
          touch "$run_dir/done"
          exit 1
        }
      grep -q "#Input tokens: ${expected_input_tokens}" "$run_dir/bench_shared100k_bs128.log" || {
        echo "Expected ${bench_num_prompts} fixed $((gsp_system_prompt_len + gsp_question_len))-token prompts" >>"$run_dir/bench_shared100k_bs128.log"
        echo FAIL >"$run_dir/status"
        touch "$run_dir/done"
        exit 1
      }
      if grep -q 'Input length (100000 tokens) exceeds' "$run_dir/rank0.log"; then
        echo "Server rejected fixed 100K input" >>"$run_dir/bench_shared100k_bs128.log"
        echo FAIL >"$run_dir/status"
        touch "$run_dir/done"
        exit 1
      fi
    fi
    echo PASS >"$run_dir/status"
  else
    echo FAIL >"$run_dir/status"
  fi
  touch "$run_dir/done"
else
  while [[ ! -e "$run_dir/done" ]]; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 2
  done
fi

if [[ "$rank" == 0 ]]; then
  echo "K3_FULL_${mode^^}_$(cat "$run_dir/status")"
  tail -160 "$run_dir/rank0.log"
fi
