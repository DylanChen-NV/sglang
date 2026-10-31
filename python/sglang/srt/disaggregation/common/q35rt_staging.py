"""Bit-preserving mixed-width KV staging for CP2 prefill -> TP1 q35rt.

Native bootstrap, allocation, notifications and completion fences remain in
NIXL. Each registered region supplies its own row width, including draft KV.
"""

import os
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl


def bulk_state_enabled():
    value = os.environ.get("SGLANG_Q35RT_BULK_STATE", "0")
    if value not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_BULK_STATE must be 0 or 1")
    return value == "1"


def configure_bulk_state(pool):
    """Describe a registered packet while keeping native compute/cache pools."""
    from q35rt.state_ops import state_packet_bytes

    cache = pool.mamba_pool.mamba_cache
    state, conv = cache.temporal, cache.conv[0]
    heads, slots = state.shape[2], state.shape[1]
    if state.shape[0] != 30 or state.dtype != torch.float32 or tuple(state.shape[-2:]) != (128, 128):
        raise ValueError("bulk state requires the validated Qwen3.5 FP32 state")
    if heads == 16:
        if conv.dtype != torch.bfloat16 or os.environ.get("SGLANG_Q35RT_PRODUCER_EVENT", "0") != "1":
            raise ValueError("bulk CP2 state requires BF16 conv and exact producer events")
    elif heads != 32 or not hasattr(pool, "q35rt_import_received"):
        raise ValueError("bulk state requires the paired CP2/q35rt TP1 deployment")
    packet = getattr(pool, "_q35rt_bulk_state", None)
    if packet is None:
        packet = torch.empty((slots, state_packet_bytes(heads)), dtype=torch.uint8, device=state.device)
        pool._q35rt_bulk_state = packet
    elif tuple(packet.shape) != (slots, state_packet_bytes(heads)):
        raise ValueError("registered bulk-state backing cannot be resized")
    return [packet.data_ptr()], [packet.nbytes], [packet.shape[1]]


def pack_bulk_state(manager, slot, producer_ready):
    from q35rt.state_ops import pack_native_state

    if producer_ready is None:
        raise RuntimeError("bulk state transfer requires this request's producer event")
    torch.cuda.set_device(manager.kv_args.gpu_id)
    if not hasattr(manager, "_q35rt_state_stream"):
        manager._q35rt_state_stream = torch.cuda.Stream()
    stream = manager._q35rt_state_stream
    stream.wait_event(producer_ready)
    cache = manager.q35rt_mamba_cache
    with torch.cuda.stream(stream):
        pack_native_state(cache.temporal, cache.conv[0], int(slot), manager.q35rt_bulk_state)
    # NIXL does not inherit PyTorch stream dependencies. Fence only this packet.
    stream.synchronize()


@triton.jit
def _pack(
    ptrs,
    widths,
    offsets,
    pages,
    output,
    num_tokens,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    region = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    width = tl.load(widths + region)
    start = tl.load(offsets + region).to(tl.int64) * num_tokens
    mask = x < num_tokens * width
    row, col = x // width, x % width
    physical = tl.load(pages + row // PAGE, mask=mask, other=0) * PAGE + row % PAGE
    source = tl.load(ptrs + region).to(tl.pointer_type(tl.uint16))
    value = tl.load(source + physical.to(tl.int64) * width + col, mask=mask, other=0)
    tl.store(output + start + x, value, mask=mask)


@triton.jit
def _scatter(
    ptrs,
    widths,
    offsets,
    pages,
    source,
    num_tokens,
    rank_elements,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    region, writer = tl.program_id(0), tl.program_id(1)
    x = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    width = tl.load(widths + region)
    mask = x < num_tokens * width
    row, col = x // width, x % width
    start = (
        writer.to(tl.int64) * rank_elements
        + tl.load(offsets + region).to(tl.int64) * num_tokens
    )
    value = tl.load(source + start + x, mask=mask, other=0)
    physical = tl.load(pages + row // PAGE, mask=mask, other=0) * PAGE + row % PAGE
    dest = tl.load(ptrs + region).to(tl.pointer_type(tl.uint16))
    tl.store(
        dest + physical.to(tl.int64) * (2 * width) + writer * width + col,
        value,
        mask=mask,
    )


def metadata(manager, receive=False):
    name = "_q35rt_receive_metadata" if receive else "_q35rt_send_metadata"
    if not hasattr(manager, name):
        args = manager.kv_args
        divisor = args.page_size * (4 if receive else 2)
        if len(args.kv_data_ptrs) != 32 or any(n % divisor for n in args.kv_item_lens):
            raise ValueError(
                "q35rt mixed staging requires native BF16 target/draft regions"
            )
        widths = [n // divisor for n in args.kv_item_lens]
        if widths != [256] * 20 + [512] * 12:
            raise ValueError(
                "q35rt mixed staging requires the validated target/draft BF16 row geometry"
            )
        offsets = np.cumsum([0] + widths[:-1]).tolist()
        device = "cuda:" + str(args.gpu_id)
        values = tuple(
            torch.tensor(v, dtype=torch.int64, device=device)
            for v in (args.kv_data_ptrs, widths, offsets)
        )
        setattr(manager, name, (values, widths, sum(widths)))
    return getattr(manager, name)


def pack(manager, pages, staging_buffer, notif, *, producer_ready=None):
    if manager.attn_tp_size != 2:
        raise ValueError("q35rt PD staging currently requires CP2 prefill")
    torch.cuda.set_device(manager.kv_args.gpu_id)
    values, widths, row_elements = metadata(manager)
    count = len(pages) * manager.kv_args.page_size
    total = count * row_elements * 2
    if not staging_buffer.fits(total):
        raise ValueError("q35rt prefill staging pool is too small")
    if not hasattr(staging_buffer, "_q35rt_stream"):
        staging_buffer._q35rt_stream = torch.cuda.Stream()
    stream = staging_buffer._q35rt_stream
    if producer_ready is None:
        stream.wait_stream(torch.cuda.default_stream())
    else:
        # This chunk's pool data is immutable until native transfer completion.
        # Later prefill work on the producer stream belongs to other requests.
        stream.wait_event(producer_ready)
    fused = os.environ.get("SGLANG_Q35RT_FUSED_PD_PACK", "0")
    if fused not in ("0", "1"):
        raise ValueError("SGLANG_Q35RT_FUSED_PD_PACK must be 0 or 1")
    with torch.cuda.stream(stream):
        indices = torch.as_tensor(np.asarray(pages, dtype=np.int64), device="cuda")
        if fused == "1":
            from q35rt.cache_ops import pack_native_cp2_pairs

            pack_native_cp2_pairs(values[0], indices, staging_buffer.buffer[:total], manager.kv_args.page_size)
        else:
            _pack[(len(widths), triton.cdiv(count * max(widths), 1024))](
                *values,
                indices,
                staging_buffer.buffer[:total].view(torch.uint16),
                count,
                manager.kv_args.page_size,
                1024
            )
    stream.synchronize()
    verify_dir = os.environ.get("Q35_PD_VERIFY_DIR")
    if verify_dir:
        root = Path(verify_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Native notification includes room, chunk and logical page start.
        parts = notif.split("_")
        torch.save(
            dict(
                widths=widths,
                tokens=count,
                page_start=int(parts[6]),
                data=staging_buffer.buffer[:total].cpu(),
            ),
            root
            / (
                parts[0]
                + "-"
                + parts[2]
                + "-"
                + str(manager.kv_args.engine_rank % 2)
                + ".pt"
            ),
        )
    return total


def scatter(manager, staging_view, pages, prefill_tp):
    if prefill_tp != 2 or manager.attn_tp_size != 1:
        raise ValueError("q35rt PD staging currently requires CP2 -> TP1")
    values, widths, row_elements = metadata(manager, receive=True)
    count = pages.numel() * manager.kv_args.page_size
    total = count * row_elements * 4
    if staging_view.numel() < total:
        raise ValueError("q35rt decode staging region is too small")
    if hasattr(manager, "q35rt_slab_targets"):
        from q35rt.cache_ops import scatter_packed_cp2_to_slabs

        scatter_packed_cp2_to_slabs(
            staging_view[:total], pages, *manager.q35rt_slab_targets,
            manager.kv_args.page_size,
        )
        return
    _scatter[(len(widths), 2, triton.cdiv(count * max(widths), 1024))](
        *values,
        pages,
        staging_view[:total].view(torch.uint16),
        count,
        count * row_elements,
        manager.kv_args.page_size,
        1024
    )


def verify_received(req, locations, buffers):
    root = os.environ.get("Q35_PD_VERIFY_DIR")
    if not root:
        return
    packets = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted(Path(root).glob(str(req.bootstrap_room) + "-0-*.pt"))
    ]
    if len(packets) != 2:
        raise AssertionError("missing actual CP2 packed KV verification packets")
    rows = locations.numel()
    offsets = [0, 0]
    for region, buffer in enumerate(buffers):
        pieces = []
        for rank, packet in enumerate(packets):
            count, width = packet["tokens"], packet["widths"][region]
            end = offsets[rank] + count * width * 2
            pieces.append(
                packet["data"][offsets[rank] : end]
                .view(torch.uint16)
                .reshape(count, width)[:rows]
            )
            offsets[rank] = end
        expected = torch.cat(pieces, dim=1)
        actual = (
            buffer[locations].contiguous().view(torch.uint16).reshape(rows, -1).cpu()
        )
        if not torch.equal(actual, expected):
            raise AssertionError(
                "actual native PD KV mismatch at region " + str(region)
            )
    print(
        "Q35_PD_KV_BITWISE_PASS room="
        + str(req.bootstrap_room)
        + " regions="
        + str(len(buffers))
        + " tokens="
        + str(rows),
        flush=True,
    )


def capture_state(manager, slot, notif):
    root = os.environ.get("Q35_PD_VERIFY_DIR")
    if not root:
        return
    state = manager.q35rt_mamba_cache
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(temporal=state.temporal[:, slot].cpu(), conv=state.conv[0][:, slot].cpu()),
        root
        / (
            notif.split("_")[0]
            + "-state-"
            + str(manager.kv_args.engine_rank % 2)
            + ".pt"
        ),
    )


def verify_state(req, temporal, conv):
    root = os.environ.get("Q35_PD_VERIFY_DIR")
    if not root:
        return
    packets = [
        torch.load(p, map_location="cpu", weights_only=False)
        for p in sorted(Path(root).glob(str(req.bootstrap_room) + "-state-*.pt"))
    ]
    if len(packets) != 2:
        raise AssertionError("missing actual CP2 Mamba verification packets")
    expected = torch.cat([p["temporal"] for p in packets], dim=1)
    assert torch.equal(
        temporal.cpu().view(torch.uint8), expected.view(torch.uint8)
    ), "PD temporal mismatch"
    expected = torch.cat(
        [
            torch.cat([p["conv"][:, start : start + width] for p in packets], dim=1)
            for start, width in ((0, 1024), (1024, 1024), (2048, 2048))
        ],
        dim=1,
    )
    assert torch.equal(
        conv.cpu().view(torch.uint8), expected.view(torch.uint8)
    ), "PD convolution mismatch"
    print("Q35_PD_STATE_BITWISE_PASS room=" + str(req.bootstrap_room), flush=True)
