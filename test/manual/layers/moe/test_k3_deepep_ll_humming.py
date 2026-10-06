#!/usr/bin/env python3
"""K3-shape Humming adapter + optional real multi-rank DeepEP LL regression.

Synthetic MXFP4 weights, not a model-accuracy benchmark. No LL GEMM imports.
Run local mode first, then torchrun --nproc_per_node=2 ... --distributed.
"""

import argparse
import inspect
import json
import math
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from flashinfer.autotuner import autotune
from flashinfer.fused_moe import preprocess_moe_weights_for_sm90_mixed_gemm_humming

from sglang.kernels.ops.quantization import sgl_per_token_quant_fp8
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

E, H, I = 28, 3584, 3072


def quantize(x):
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    sf = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    sgl_per_token_quant_fp8(x, q, sf)
    return q, sf


def make_weights(seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def prepare(n, k):
        packed = torch.empty((E, n, k // 2), device="cuda", dtype=torch.int8)
        packed.random_(-128, 127, generator=gen)
        scales = torch.full((E, n, k // 32), 127, device="cuda", dtype=torch.uint8)
        outputs = [[], [], []]
        for begin in range(0, E, 7):
            values = preprocess_moe_weights_for_sm90_mixed_gemm_humming(
                packed[begin:begin + 7].view(torch.uint8).contiguous(),
                scales[begin:begin + 7].contiguous(),
            )
            for parts, value in zip(outputs, values):
                parts.append(value)
        return [torch.cat(parts).contiguous() for parts in outputs]

    w13, s13, r13 = prepare(2 * I, H)
    w2, s2, r2 = prepare(H, I)
    return runner.FlashInferCutlassMxfp4MoeQuantInfo(
        w13, w2, s13, s2,
        w13_humming_residual_scale=(r13 * 64).contiguous(),
        w2_humming_residual_scale=(r2 * 64).contiguous(),
        humming_fc2_act_scale=torch.ones((), device="cuda", dtype=torch.float32),
        use_situ=True,
        situ_beta=torch.full((E,), 4.0, device="cuda"),
        situ_linear_beta=torch.full((E,), 25.0, device="cuda"),
    )


CONFIG = MoeRunnerConfig(top_k=1, activation="situ", gemm1_alpha=4.0, gemm1_clamp_limit=25.0)


def standard(x, ids, weights, qi, sf=None):
    dispatch = StandardDispatchOutput(x, sf, StandardTopKOutput(weights, ids, None))
    return runner.fused_experts_none_to_flashinfer_mxfp4(dispatch, qi, CONFIG).hidden_states


def numerical_metrics(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("non-finite result")
    rel = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None]))
    return {"relative_l2": rel, "cosine": cosine, "max_abs": float((a - b).abs().max())}


def errors(actual, reference):
    metrics = numerical_metrics(actual, reference)
    print("NUMERICS", metrics, flush=True)
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0.025, atol=0.025)
    assert metrics["relative_l2"] < 0.01, metrics
    return metrics


def graph_time(fn, tests):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    samples = []
    for _ in range(3):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(tests):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / tests)
    return out, statistics.median(samples)


def graph_components(pipeline, tests):
    """Separate instrumented graph; clean pipeline timing stays unmodified."""
    marks = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(4)]
    pipeline(marks)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = pipeline(marks)
    samples = []
    for _ in range(3):
        for _ in range(tests):
            graph.replay()
        torch.cuda.synchronize()
        samples.append([marks[i].elapsed_time(marks[i + 1]) * 1000 for i in range(3)])
    return out, dict(zip(
        ("dispatch_hook_us", "adapter_humming_us", "combine_hook_us"),
        [statistics.median([sample[i] for sample in samples]) for i in range(3)],
    ))


def local_tests(qi, args):
    records = []
    for capacity, active, count in ((8, 8, 8), (4, 16, 4), (2, 1, 1), (4, 0, 0)):
        counts = torch.zeros(E, device="cuda", dtype=torch.int32)
        counts[:active] = count
        x = torch.randn(E, capacity, H, device="cuda", dtype=torch.bfloat16).mul_(0.05)
        valid = torch.arange(capacity, device="cuda")[None, :] < counts[:, None]
        x[~valid] = float("nan")
        # Host extraction below is REFERENCE SETUP ONLY, never adapter runtime.
        ref_ids = torch.arange(E, device="cuda")[:, None].expand(E, capacity)[valid, None].int()
        ref_x = x[valid].contiguous()
        ref_weights = torch.ones((ref_x.shape[0], 1), device="cuda")
        for carrier in ("bf16", "fp8"):
            if carrier == "fp8":
                flat = x.nan_to_num().view(-1, H)
                q, scale = quantize(flat)
                recv = q.view_as(x)
                recv_sf = scale.view(E, capacity, 1).expand(E, capacity, H // 128)
                ref_q = recv.float()[valid].to(torch.float8_e4m3fn).contiguous()
                ref_sf = recv_sf[..., 0][valid, None].contiguous()
            else:
                recv, recv_sf = x, None
            d = DeepEPLLDispatchOutput(
                recv, recv_sf, torch.tensor([[0]], device="cuda", dtype=torch.int64),
                torch.ones((1, 1), device="cuda"), counts, count,
            )

            def fn():
                return runner.fused_experts_deepep_to_flashinfer_mxfp4(d, qi, CONFIG).hidden_states

            # Tune both tensor shapes outside graph capture and timing.
            with autotune(tune_mode=True):
                fn()
                if active:
                    ref = standard(ref_q if carrier == "fp8" else ref_x,
                                   ref_ids, ref_weights, qi, ref_sf if carrier == "fp8" else None)
            result, latency = graph_time(fn, args.tests)
            torch.cuda.synchronize()
            err = errors(result[valid], ref) if active else {"zero_output": bool((result == 0).all())}
            assert result[~valid].abs().sum() == 0
            assert active or err["zero_output"]
            records.append({"carrier": carrier, "capacity": capacity, "active_experts": active,
                            "valid_rows": active * count, "padded_rows": E * capacity,
                            "adapter_humming_graph_us": latency, **err})
            print("LOCAL", records[-1], flush=True)
    return records


def distributed_tests(qi, args, rank, world):
    from deep_ep import Buffer

    # Pinned DeepEP legacy kernel aligns each expert's FP8 scale base to
    # 128 receive slots, while its allocation uses the requested capacity.
    # Use a matching capacity: smaller non-aligned carriers overwrite scales
    # and can write outside their allocation (not a Humming numerical issue).
    quantum = 128 // math.gcd(world, 128)
    max_send = ((max(args.batches) + quantum - 1) // quantum) * quantum
    global_experts = E * world
    rdma_bytes = Buffer.get_low_latency_rdma_size_hint(max_send, H, world, global_experts)
    buffer = Buffer(dist.group.WORLD, num_rdma_bytes=rdma_bytes, low_latency_mode=True,
                    num_qps_per_rank=E)
    assert "use_per_token_scale" in inspect.signature(buffer.low_latency_dispatch).parameters
    records = []
    for bs in args.batches:
        for routing in ("spread", "hot"):
            topk = 4
            x = torch.randn(bs, H, device="cuda", dtype=torch.bfloat16).mul_(0.05)
            ids = torch.arange(bs * topk, device="cuda").reshape(bs, topk)
            ids = (ids + rank * 11) % global_experts if routing == "spread" else ids % topk
            ids = ids.to(torch.int64)
            weights = torch.rand(bs, topk, device="cuda")
            weights /= weights.sum(-1, keepdim=True)
            for carrier in ("bf16", "fp8"):
                send = quantize(x) if carrier == "fp8" else x

                def pipeline(marks=None):
                    if marks is not None:
                        marks[0].record()
                    recv, counts, handle, event, hook = buffer.low_latency_dispatch(
                        send, ids, max_send, global_experts,
                        use_fp8=carrier == "fp8", use_per_token_scale=carrier == "fp8",
                        async_finish=False, return_recv_hook=True,
                    )
                    hook()
                    if marks is not None:
                        marks[1].record()
                    recv_x, recv_sf = recv if isinstance(recv, tuple) else (recv, None)
                    d = DeepEPLLDispatchOutput(recv_x, recv_sf, ids, weights, counts, 1)
                    out = runner.fused_experts_deepep_to_flashinfer_mxfp4(d, qi, CONFIG).hidden_states
                    if marks is not None:
                        marks[2].record()
                    combined, event, hook = buffer.low_latency_combine(
                        out, ids, weights, handle, async_finish=False, return_recv_hook=True,
                    )
                    hook()
                    if marks is not None:
                        marks[3].record()
                    return combined

                # Compute a Standard EP reference over all input tokens, with
                # local IDs and a reduction over expert shards. Same FP8 bytes.
                def gather(tensor):
                    xs = [torch.empty_like(tensor) for _ in range(world)]
                    dist.all_gather(xs, tensor)
                    return torch.cat(xs)

                all_ids, all_weights = gather(ids), gather(weights)
                if carrier == "fp8":
                    all_x = gather(send[0].view(torch.uint8)).view(torch.float8_e4m3fn)
                    all_sf = gather(send[1])
                else:
                    all_x, all_sf = gather(x), None
                local_ids = all_ids - rank * E
                local_ids.masked_fill_((local_ids < 0) | (local_ids >= E), -1)
                with autotune(tune_mode=True):
                    direct_ref = standard(all_x, local_ids, all_weights, qi, all_sf)
                    # Match DeepEP combine's precision boundary: each expert
                    # FC2 result is BF16, weights and accumulation are FP32,
                    # and the final token sum is cast once. Standard EP rounds
                    # a partial top-k sum on each shard BEFORE all-reduce.
                    assignment_out = standard(
                        all_x.repeat_interleave(topk, dim=0),
                        local_ids.reshape(-1, 1).int(),
                        torch.ones_like(all_weights.reshape(-1, 1)), qi,
                        all_sf.repeat_interleave(topk, dim=0) if all_sf is not None else None,
                    )
                dist.all_reduce(direct_ref)
                direct_ref = direct_ref[rank * bs:(rank + 1) * bs].clone()
                ref = (assignment_out.float().view(world * bs, topk, H)
                       * all_weights[..., None]).sum(1)
                dist.all_reduce(ref)
                ref = ref[rank * bs:(rank + 1) * bs].to(torch.bfloat16).contiguous()
                # Rank-varying Humming autotune must not occur inside a live
                # DeepEP collective; the full padded top-k1 shape is warmed
                # separately with synthetic counts before entering the pipe.
                c = max_send * world
                dummy_x = torch.zeros((E, c, H), dtype=torch.bfloat16, device="cuda")
                dummy_sf = None
                if carrier == "fp8":
                    dummy_q, dummy_scale = quantize(dummy_x.view(-1, H))
                    dummy_x = dummy_q.view(E, c, H)
                    dummy_sf = dummy_scale.view(E, c, 1).expand(E, c, H // 128)
                dummy = DeepEPLLDispatchOutput(dummy_x, dummy_sf, ids, weights,
                                              torch.full((E,), min(4, c), device="cuda", dtype=torch.int32), 4)
                with autotune(tune_mode=True):
                    runner.fused_experts_deepep_to_flashinfer_mxfp4(dummy, qi, CONFIG)
                dist.barrier()
                out = pipeline()
                torch.cuda.synchronize()
                print("CASE", rank, bs, routing, carrier, flush=True)
                if carrier == "fp8" and args.diagnose:
                    # Diagnostic reference with one physical row per assignment:
                    # distinguish top-k>1 prequantized Humming scale indexing
                    # from transport/adapter issues, without relaxing the gate.
                    expanded_x = all_x.repeat_interleave(topk, dim=0)
                    expanded_sf = all_sf.repeat_interleave(topk, dim=0)
                    expanded_ids = local_ids.reshape(-1, 1).int()
                    with autotune(tune_mode=True):
                        per_assignment = standard(
                            expanded_x, expanded_ids,
                            torch.ones_like(all_weights.reshape(-1, 1)), qi, expanded_sf,
                        )
                    explicit = (per_assignment.float().view(world * bs, topk, H)
                                * all_weights[..., None]).sum(1)
                    dist.all_reduce(explicit)
                    explicit = explicit[rank * bs:(rank + 1) * bs].contiguous()
                    print("EXPLICIT_REFERENCE", rank,
                          float(torch.linalg.vector_norm(out.float() - explicit)
                                / torch.linalg.vector_norm(explicit).clamp_min(1e-12)),
                          "STANDARD_REFERENCE", float(torch.linalg.vector_norm(ref.float() - explicit)
                                / torch.linalg.vector_norm(explicit).clamp_min(1e-12)), flush=True)
                    # DeepEP result buffers are ping-pong aliases: inspect a
                    # fresh dispatch and match each received row to an input.
                    recv, counts, _, _, recv_hook = buffer.low_latency_dispatch(
                        send, ids, max_send, global_experts, use_fp8=True,
                        use_per_token_scale=True, return_recv_hook=True,
                    )
                    recv_hook()
                    rx, rs = recv
                    diagnostics = []
                    for expert in range(E):
                        n = int(counts[expert])
                        candidates = ((all_ids == expert + rank * E).any(-1)).nonzero().flatten()
                        for row in range(n):
                            matches = (rx[expert, row].view(torch.uint8)[None, :]
                                       == all_x.view(torch.uint8)[candidates]).all(-1)
                            matched = candidates[matches]
                            diagnostics.append({"expert": expert, "row": row,
                                "matched_tokens": matched.tolist(),
                                "scale": float(rs[expert, row, 0]),
                                "expected_scales": all_sf[matched].flatten().tolist(),
                                "all_groups_same": bool((rs[expert, row] == rs[expert, row, 0]).all())})
                    print("TRANSPORT_DIAGNOSTICS", rank, diagnostics, flush=True)
                    dump = args.output.with_name("diagnostic.rank%d.bs%d.%s.pt" % (rank, bs, routing))
                    torch.save({"actual": out.clone(), "standard": ref.clone(),
                                "explicit": explicit, "transport": diagnostics}, dump)
                eager_err = errors(out, ref)
                dist.barrier()
                out, us = graph_time(pipeline, args.tests)
                torch.cuda.synchronize()
                graph_err = errors(out, ref)
                components = {}
                if args.components:
                    dist.barrier()
                    profiled, components = graph_components(pipeline, args.tests)
                    errors(profiled, ref)
                    if carrier == "fp8":
                        _, components["predispatch_quant_graph_us"] = graph_time(
                            lambda: quantize(x), args.tests
                        )
                records.append({"carrier": carrier, "batch_per_rank": bs,
                                "routing": routing, "rank": rank, "world": world,
                                "capacity_per_expert": c, "pipeline_graph_us": us,
                                "eager": eager_err, "graph": graph_err,
                                "standard_ep_partial_bf16_vs_oracle": numerical_metrics(direct_ref, ref),
                                "instrumented_components": components})
                print("DISTRIBUTED", records[-1], flush=True)
                dist.barrier()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--tests", type=int, default=30)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--components", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.batches = [int(x) for x in args.batches.split(",")]
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    name = torch.cuda.get_device_name(rank)
    assert torch.cuda.get_device_capability(rank) == (9, 0), name
    torch.manual_seed(20261006 + rank)
    if args.distributed:
        dist.init_process_group("nccl")
    qi = make_weights(20261006 + rank)
    # Only allocator bookkeeping is stubbed for an isolated kernel test.
    # GEMM/quant/routing/dispatch/combine implementations are real.
    with patch.object(runner, "get_tp_group", return_value=None), \
         patch.object(runner, "is_allocation_symmetric", return_value=False), \
         patch.object(runner, "use_symmetric_memory", side_effect=lambda *a, **k: nullcontext()):
        records = distributed_tests(qi, args, rank, world) if args.distributed else local_tests(qi, args)
    import flashinfer
    payload = {"status": "PASS", "hardware": name, "rank": rank, "world": world,
               "shape": {"experts_per_rank": E, "hidden": H, "intermediate": I},
               "scope": "synthetic K3-shape MoE, not full-model accuracy or E2E",
               "timing_scope": "clean graph: dispatch+adapter+Humming+combine; FP8 send quant excluded; components use separate external-event graph",
               "flashinfer_file": flashinfer.__file__, "flashinfer_version": flashinfer.__version__,
               "torch_version": torch.__version__, "timestamp": time.time(), "results": records}
    output = args.output.with_name(args.output.stem + (".rank%d" % rank if args.distributed else "") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
