#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  b1|f1) ;;
  *) echo "usage: $0 {b1|f1}" >&2; exit 2 ;;
esac

base=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/kimi-k3-deepep-lowlatency/b1-validation
model=/lustre/fs1/portfolios/coreai/projects/coreai_devtech_all/users/ziqingc/05_claude_ws/models/Kimi-K3-official-f831ab6
run_id="${K3_RUN_ID:?K3_RUN_ID must match the PD role processes}"
run_root="$base/results/pd-k3-b1-f1-${run_id}"
bench_dir="$run_root/bench_${mode}"
decode_role="decode_${mode}"
mkdir -p "$bench_dir"

sglang_src="${K3_SGLANG_SRC:-$base/sglang}"
export PYTHONPATH="$sglang_src/python:${PYTHONPATH:-}"
prefill_host=$(head -1 "$run_root/prefill_master_host")
decode_host=$(head -1 "$run_root/decode_master_host")

for marker in ready_prefill "ready_${decode_role}"; do
  for _ in $(seq 1 1800); do
    [[ -e "$run_root/$marker" ]] && break
    [[ -e "$run_root/failed_prefill" || -e "$run_root/failed_${decode_role}" ]] && exit 1
    sleep 2
  done
  [[ -e "$run_root/$marker" ]] || exit 1
done

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${prefill_host}:30000" 8998 \
  --decode "http://${decode_host}:30100" \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-circuit-breaker \
  --health-check-interval-secs 999999 \
  >"$bench_dir/router.log" 2>&1 &
router_pid=$!
cleanup() {
  kill "$router_pid" 2>/dev/null || true
  wait "$router_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 300); do
  curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  kill -0 "$router_pid" 2>/dev/null || exit 1
  sleep 1
done

python3 -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --dataset-name generated-shared-prefix \
  --tokenizer "$model" \
  --model "$model" \
  --num-prompts 128 \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 128 \
  --gsp-system-prompt-len 99999 \
  --gsp-question-len 1 \
  --gsp-output-len 128 \
  --gsp-range-ratio 1.0 \
  --gsp-fast-prepare \
  --gsp-ordered \
  --gsp-send-routing-key \
  --max-concurrency 128 \
  --request-rate inf \
  --warmup-requests 1 \
  --seed 42 \
  --output-details \
  --output-file "$bench_dir/bench_shared100k_bs128.jsonl" \
  >"$bench_dir/bench_shared100k_bs128.log" 2>&1

grep -q '#Input tokens: 12800000' "$bench_dir/bench_shared100k_bs128.log"
grep -q 'Successful requests:                     128' "$bench_dir/bench_shared100k_bs128.log"
echo PASS >"$bench_dir/status"

