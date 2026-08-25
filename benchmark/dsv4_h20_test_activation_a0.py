import json
import os
import statistics
import sys

os.environ.setdefault("HOME", "/work/home-a0")
os.environ.setdefault("XDG_CACHE_HOME", "/work/xdg-a0")
os.environ.setdefault("SGLANG_CACHE_DIR", "/work/cache/sglang-a0")
sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch
from humming import ops as hops

from sglang.kernels.ops.moe.fused_activation_quant import fused_swiglu_quant_fp8
from sglang.kernels.ops.moe.fused_moe_triton_kernels import act_and_mul_triton

E, I, K, LIMIT = 256, 512, 6, 10.0
torch.cuda.set_device(0)
torch.manual_seed(20260825)


def graph_latency(fn, warmup=10, iters=1000, repeats=5):
    for _ in range(warmup):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iters)
    return statistics.median(samples), samples


results = []
for bs in (2, 4, 6, 8, 10):
    rows = bs * K
    counts = torch.zeros(E, dtype=torch.int32, device="cuda")
    chosen = torch.randperm(E, device="cuda")[:rows]
    counts.scatter_add_(0, chosen, torch.ones_like(chosen, dtype=torch.int32))
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device="cuda"), counts.cumsum(0)]
    ).to(torch.int32)
    row_experts = torch.repeat_interleave(
        torch.arange(E, device="cuda"), counts.long()
    )
    residual = torch.rand(E, dtype=torch.float32, device="cuda") + 0.5
    gate_up = (torch.randn(rows, 2 * I, device="cuda") * 3).to(torch.bfloat16)
    activated = torch.empty(rows, I, dtype=torch.bfloat16, device="cuda")
    ref_q = torch.empty(rows, I, dtype=torch.float8_e4m3fn, device="cuda")
    fused_q = torch.empty_like(ref_q)
    fused_scales = torch.empty(rows, 1, dtype=torch.float32, device="cuda")

    def reference():
        act_and_mul_triton(
            gate_up, activated, {}, activation="silu", swiglu_limit=LIMIT
        )
        return hops.quant_input(activated, "float8e4m3", outputs=ref_q)

    def fused():
        return fused_swiglu_quant_fp8(
            gate_up,
            offsets,
            residual,
            LIMIT,
            outputs=fused_q,
            scales=fused_scales,
        )

    _, reference_scales = reference()
    fused()
    expected_scales = (
        reference_scales.flatten() * residual[row_experts] * 64.0
    ).reshape_as(fused_scales)
    torch.cuda.synchronize()
    ref_us, ref_samples = graph_latency(reference)
    fused_us, fused_samples = graph_latency(fused)
    row = {
        "bs": bs,
        "rows": rows,
        "quant_exact": bool(torch.equal(ref_q, fused_q)),
        "scale_max_abs": float((expected_scales - fused_scales).abs().max()),
        "scale_max_rel": float(
            (
                (expected_scales - fused_scales).abs()
                / expected_scales.abs().clamp_min(1.0e-30)
            ).max()
        ),
        "reference_activation_quant_us": ref_us,
        "fused_activation_quant_combine_us": fused_us,
        "reference_samples_us": ref_samples,
        "fused_samples_us": fused_samples,
    }
    if not row["quant_exact"] or row["scale_max_rel"] > 1.0e-6:
        raise AssertionError(row)
    results.append(row)
    print(json.dumps({"phase": "result", **row}), flush=True)

with open("/work/results/a0_correctness.json", "w") as handle:
    json.dump({"results": results}, handle, indent=2)
print(json.dumps({"phase": "done"}), flush=True)
