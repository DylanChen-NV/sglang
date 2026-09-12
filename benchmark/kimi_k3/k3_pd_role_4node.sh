#!/usr/bin/env bash
set -euo pipefail

role="${1:-}"
case "$role" in
  prefill|decode_b1|decode_f1) ;;
  *) echo "usage: $0 {prefill|decode_b1|decode_f1}" >&2; exit 2 ;;
esac

base=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/kimi-k3-deepep-lowlatency/b1-validation
model=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/models/Kimi-K3-official-f831ab6
run_id="${K3_RUN_ID:?K3_RUN_ID must identify the shared PD run}"
run_root="$base/results/pd-k3-b1-f1-${run_id}"
role_dir="$run_root/$role"
local_rank="${RANK:-${SLURM_NODEID:-0}}"
rank="$((local_rank + ${K3_NODE_RANK_OFFSET:-0}))"
mkdir -p "$role_dir"

sglang_src="${K3_SGLANG_SRC:-$base/sglang}"
llgg_src="${K3_LLGG_SRC:-$base/LowLatencyGroupedGEMM}"
llgg_build="${K3_LLGG_BUILD:-$base/build/lowlatency-extension-c40c108}"
export PYTHONPATH="$sglang_src/python:$llgg_build:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export SGLANG_WARMUP_TIMEOUT=1800
export SGLANG_CACHE_DIR="$base/cache/sglang"
export FLASHINFER_WORKSPACE_DIR="$base/cache/flashinfer"
transfer_backend="${K3_PD_TRANSFER_BACKEND:-mooncake}"
ib_devices="${K3_PD_IB_DEVICES:-mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8}"
# NIXL UCX does not honor --disaggregation-ib-device. DFW mlx5_0 cannot register these CUDA buffers.
export UCX_NET_DEVICES="${K3_UCX_NET_DEVICES:-mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1}"

node_cache="/tmp/k3-pd-${run_id}-${role}-$(hostname)"
mkdir -p "$node_cache"
export XDG_CACHE_HOME="$node_cache/xdg"
export TORCHINDUCTOR_CACHE_DIR="$node_cache/torchinductor"
export TRITON_CACHE_DIR="$node_cache/triton"
export SGLANG_JIT_CACHE_DIR="$node_cache/sglang-jit"

python3 -c 'import torch; assert torch.cuda.device_count() == 8; assert all(torch.cuda.get_device_name(i).startswith("NVIDIA H100") for i in range(8)); [torch.empty(1, device=f"cuda:{i}") for i in range(8)]' >"$role_dir/gpu_probe_rank${rank}.log" 2>&1

if [[ "$role" == prefill ]]; then
  host_file="$run_root/prefill_master_host"
  port=30000
  dist_port=20000
  mem_fraction_static="${K3_PREFILL_MEM_FRACTION_STATIC:-0.84}"
  role_args=(
    --disaggregation-mode prefill
    --disaggregation-transfer-backend "$transfer_backend"
    --disaggregation-ib-device "$ib_devices"
    --disaggregation-bootstrap-port 8998
    --moe-runner-backend flashinfer_mxfp4
    --flashinfer-mxfp4-moe-precision fp8
    --moe-a2a-backend deepep
    --deepep-mode normal
    --deepep-dispatcher-output-dtype bf16
    --cuda-graph-backend-decode disabled
  )
else
  host_file="$run_root/decode_master_host"
  port=30100
  dist_port=21000
  mem_fraction_static="${K3_DECODE_MEM_FRACTION_STATIC:-0.841}"
  decode_extra_slots="${K3_PD_DECODE_EXTRA_SLOTS:-64}"
  decode_radix_cache="${K3_PD_DECODE_RADIX_CACHE:-0}"
  export SGLANG_LOWLATENCY_DEEPEP_LAYOUT=compact
  export SGLANG_LOWLATENCY_MXFP4_VARIANT=final
  export SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS="${SGLANG_LOWLATENCY_MXFP4_PERSISTENT_CTAS:-528}"
  if [[ "$role" == decode_b1 ]]; then
    deepep_dtype=bf16
  else
    deepep_dtype=fp8
  fi
  role_args=(
    --disaggregation-mode decode
    --disaggregation-transfer-backend "$transfer_backend"
    --disaggregation-ib-device "$ib_devices"
    --disaggregation-bootstrap-port 8998
    --disaggregation-decode-extra-slots "$decode_extra_slots"
    --num-reserved-decode-tokens 128
    --moe-runner-backend lowlatency_mxfp4
    --moe-a2a-backend deepep
    --deepep-mode low_latency
    --deepep-dispatcher-output-dtype "$deepep_dtype"
    --cuda-graph-backend-decode full
    --cuda-graph-max-bs-decode 128
    --cuda-graph-bs-decode 128
  )
  if [[ "$decode_radix_cache" == 1 ]]; then
    role_args+=(--disaggregation-decode-enable-radix-cache)
  fi
fi

if [[ "$rank" == 0 ]]; then
  hostname >"$host_file"
  {
    echo "role=$role"
    echo "run_id=$run_id"
    echo "slurm_job_id=${SLURM_JOB_ID:-persistent}"
    echo "sglang_sha=$(git -C "$sglang_src" rev-parse HEAD)"
    echo "llgg_sha=$(git -C "$llgg_src" rev-parse HEAD)"
    echo "llgg_build=$llgg_build"
  } >"$role_dir/code_manifest.txt"
fi
for _ in $(seq 1 300); do
  [[ -s "$host_file" ]] && break
  sleep 1
done
master_host=$(head -1 "$host_file")

python3 -m sglang.launch_server \
  --model-path "$model" \
  --tokenizer-path "$model" \
  --trust-remote-code \
  --random-seed 42 \
  --language-only \
  --tp-size 32 \
  --ep-size 32 \
  --nnodes 4 \
  --node-rank "$rank" \
  --dist-init-addr "${master_host}:${dist_port}" \
  --prefill-attention-backend fa3 \
  --decode-attention-backend flashmla \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 131072 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests 128 \
  --max-mamba-cache-size 640 \
  --mamba-ssm-dtype bfloat16 \
  --enable-shared-experts-attn-tp \
  --mem-fraction-static "$mem_fraction_static" \
  --cuda-graph-backend-prefill disabled \
  --skip-server-warmup \
  --watchdog-timeout 3600 \
  --host 0.0.0.0 \
  --port "$port" \
  "${role_args[@]}" \
  >"$role_dir/rank${rank}.log" 2>&1 &
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
    echo PASS >"$role_dir/status"
    touch "$run_root/ready_${role}"
  else
    echo FAIL >"$role_dir/status"
    touch "$run_root/failed_${role}"
  fi
fi

while [[ ! -e "$run_root/stop_${role}" ]]; do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 2
done

if [[ -e "$run_root/failed_${role}" ]]; then
  exit 1
fi

