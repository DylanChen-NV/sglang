from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import statistics
from pathlib import Path
from typing import Callable

import torch

from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)

_CACHE_VERSION = 1
_DEFAULT_KERNEL_VERSION = "91b5461"
_PROCESS_CACHE: dict[str, int] = {}


def _cache_path() -> Path:
    configured = os.getenv("SGLANG_LOWLATENCY_MXFP4_AUTOTUNE_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_root = Path(
        os.getenv(
            "SGLANG_CACHE_DIR",
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "sglang",
        )
    )
    return cache_root / "lowlatency_mxfp4_autotune.json"


def _key(*, variant: str, rows: int, hidden_size: int, intermediate_size: int):
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    payload = {
        "gpu": properties.name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": properties.multi_processor_count,
        "cuda": torch.version.cuda,
        "kernel": os.getenv(
            "SGLANG_LOWLATENCY_MXFP4_KERNEL_VERSION", _DEFAULT_KERNEL_VERSION
        ),
        "variant": variant,
        "rows": rows,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid LowLatency autotune cache %s: %s", path, exc)
        return {}
    if payload.get("version") != _CACHE_VERSION:
        return {}
    entries = {}
    for key, value in payload.get("entries", {}).items():
        if isinstance(value, dict):
            entries[str(key)] = value
        else:
            entries[str(key)] = {"persistent_ctas": int(value)}
    return entries


def _write_cache(
    path: Path, key: str, value: int, timings: dict[int, float]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        entries = _read_cache(path)
        entries[key] = {
            "persistent_ctas": value,
            "timings_us": {
                str(candidate): elapsed for candidate, elapsed in timings.items()
            },
        }
        payload = {"version": _CACHE_VERSION, "entries": entries}
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _candidates(default: int) -> list[int]:
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm_count = properties.multi_processor_count
    multiples = (sm_count * multiplier for multiplier in range(1, 7))
    return sorted({default, 528, *multiples})


def _measure(run_candidate: Callable[[int], object], candidate: int) -> float:
    for _ in range(2):
        run_candidate(candidate)
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_candidate(candidate)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def select_persistent_ctas(
    *,
    variant: str,
    rows: int,
    hidden_size: int,
    intermediate_size: int,
    default: int,
    run_candidate: Callable[[int], object],
) -> int:
    """Select a LowLatency tactic before graph capture and persist the result."""

    mode = os.getenv("SGLANG_LOWLATENCY_MXFP4_AUTOTUNE_MODE", "auto").lower()
    if mode not in {"auto", "force", "off", "readonly"}:
        raise ValueError(
            "SGLANG_LOWLATENCY_MXFP4_AUTOTUNE_MODE must be auto, force, off, "
            f"or readonly; got {mode!r}."
        )
    if mode == "off" or rows > 64:
        return default

    key = _key(
        variant=variant,
        rows=rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if key in _PROCESS_CACHE:
        return _PROCESS_CACHE[key]

    path = _cache_path()
    if mode != "force":
        cached_entry = _read_cache(path).get(key)
        if cached_entry is not None:
            cached = int(cached_entry["persistent_ctas"])
            _PROCESS_CACHE[key] = cached
            log_info_on_rank0(
                logger,
                f"LowLatency autotune cache hit rows={rows} variant={variant} "
                f"persistent_ctas={cached}",
            )
            return cached

    if mode == "readonly":
        raise RuntimeError(
            "LowLatency autotune readonly cache miss for "
            f"rows={rows}, variant={variant}, cache={path}."
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "LowLatency autotune cache miss during CUDA Graph capture; run warmup "
            "with autotune mode auto/force before capture."
        )

    timings = {}
    for candidate in _candidates(default):
        try:
            elapsed_us = _measure(run_candidate, candidate)
        except Exception as exc:
            logger.warning(
                "LowLatency autotune candidate failed rows=%d variant=%s "
                "persistent_ctas=%d: %s",
                rows,
                variant,
                candidate,
                exc,
            )
            continue
        if math.isfinite(elapsed_us) and elapsed_us > 0:
            timings[candidate] = elapsed_us

    if not timings:
        raise RuntimeError(
            f"All LowLatency autotune candidates failed for rows={rows}, "
            f"variant={variant}."
        )
    selected = min(timings, key=timings.get)
    _PROCESS_CACHE[key] = selected
    _write_cache(path, key, selected, timings)
    log_info_on_rank0(
        logger,
        f"LowLatency autotune selected rows={rows} variant={variant} "
        f"persistent_ctas={selected} timings_us={timings} cache={path}",
    )
    return selected
