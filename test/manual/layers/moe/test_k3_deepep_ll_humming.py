#!/usr/bin/env python3
"""K3-shape Humming adapter + optional real multi-rank DeepEP LL regression.

Synthetic MXFP4 weights, not a model-accuracy benchmark. No LL GEMM imports.
Run local mode first, then torchrun --nproc_per_node=2 ... --distributed.
"""

import argparse
import inspect
import json
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


def errors(actual, reference):
    a, b = actual.float().flatten(), reference.float().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("non-finite result")
    rel = float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(a[None], b[None]))
    torch.testing.assert_close(actual.float(), reference.float(), rtol=0.025, atol=0.025)
    assert rel < 0.01, (rel, cosine)
    return {"relative_l2": rel, "cosine": cosine, "max_abs": float((a - b).abs().max())}


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

    max_send = max(args.batches)
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

                def pipeline():
                    recv, counts, handle, event, hook = buffer.low_latency_dispatch(
                        send, ids, max_send, global_experts,
                        use_fp8=carrier == "fp8", use_per_token_scale=carrier == "fp8",
                        async_finish=False, return_recv_hook=True,
                    )
                    hook()
                    recv_x, recv_sf = recv if isinstance(recv, tuple) else (recv, None)
                    d = DeepEPLLDispatchOutput(recv_x, recv_sf, ids, weights, counts, 1)
                    out = runner.fused_experts_deepep_to_flashinfer_mxfp4(d, qi, CONFIG).hidden_states
                    combined, event, hook = buffer.low_latency_combine(
                        out, ids, weights, handle, async_finish=False, return_recv_hook=True,
                    )
                    hook()
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
                    ref = standard(all_x, local_ids, all_weights, qi, all_sf)
                dist.all_reduce(ref)
                ref = ref[rank * bs:(rank + 1) * bs].contiguous()
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
                eager_err = errors(out, ref)
                dist.barrier()
                out, us = graph_time(pipeline, args.tests)
                torch.cuda.synchronize()
                graph_err = errors(out, ref)
                records.append({"carrier": carrier, "batch_per_rank": bs,
                                "routing": routing, "rank": rank, "world": world,
                                "capacity_per_expert": c, "pipeline_graph_us": us,
                                "eager": eager_err, "graph": graph_err})
                print("DISTRIBUTED", records[-1], flush=True)
                dist.barrier()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--tests", type=int, default=30)
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
               "flashinfer_file": flashinfer.__file__, "flashinfer_version": flashinfer.__version__,
               "torch_version": torch.__version__, "timestamp": time.time(), "results": records}
    output = args.output.with_name(args.output.stem + (".rank%d" % rank if args.distributed else "") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
