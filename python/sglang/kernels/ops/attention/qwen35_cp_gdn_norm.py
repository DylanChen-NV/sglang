# SPDX-License-Identifier: Apache-2.0
"""Qwen35 GDN split with the pinned native L2 normalization arithmetic.

The native normalization uses a 16x128 tile, FP32 squared sum and division by
sqrt(sum + 1e-6), then a BF16 store. Keep that tile and arithmetic; write the
separate Q/K outputs directly and copy V as integer bits in the same program.
"""

import functools
import hashlib
from pathlib import Path

import torch
import triton
import triton.language as tl


@functools.lru_cache(maxsize=1)
def _check_reference():
    source = Path(__file__).parent / "fla/l2norm.py"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != "b1323047a7c7f46268a9c295f4fea5c53a210ebfb04db5b17d7ce6ff4b24b7f7"
    ):
        raise RuntimeError("Native GDN L2 normalization changed; revalidate CP fusion")


@triton.jit
def _split_norm(X, Q, K, V, ST: tl.constexpr, SD: tl.constexpr):
    token, family = tl.program_id(0), tl.program_id(1)
    ptr = tl.make_block_ptr(
        X + token * ST + family * 2048 * SD,
        (16, 128),
        (128 * SD, SD),
        (0, 0),
        (16, 128),
        (1, 0),
    )
    x = tl.load(ptr).to(tl.float32)
    variance = tl.sum(x * x, axis=1)
    normalized = x / tl.sqrt(variance + 1e-6)[:, None]
    output = tl.where(family == 0, Q, K) + token * 2048
    out = tl.make_block_ptr(
        output, (16, 128), (128, 1), (0, 0), (16, 128), (1, 0)
    )
    tl.store(out, normalized.to(out.dtype.element_ty))
    head = tl.arange(0, 16)
    col = tl.arange(0, 128)
    offsets = head[:, None] * 128 + col[None, :]
    raw = tl.load(
        X.to(tl.pointer_type(tl.int16))
        + token * ST
        + (4096 + family * 2048 + offsets) * SD
    )
    tl.store(
        V.to(tl.pointer_type(tl.int16)) + token * 4096 + family * 2048 + offsets,
        raw,
    )


def split_norm(x):
    _check_reference()
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16:
        raise ValueError("CP GDN split requires the packed Qwen35 BF16 layout")
    n = len(x)
    q = torch.empty((n, 16, 128), dtype=x.dtype, device=x.device)
    k = torch.empty_like(q)
    v = torch.empty((n, 32, 128), dtype=x.dtype, device=x.device)
    if n:
        _split_norm[(n, 2)](
            x, q, k, v, x.stride(0), x.stride(1), num_warps=8, num_stages=3
        )
    return q, k, v
