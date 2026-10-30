"""Optional SM100 FP32 CP affine kernel using pinned CUTLASS SIMT headers.

The model validates BF16 O, BF16-rounded FP32 U and FP32 S before calling this
internal launcher. CUDA launches use the current PyTorch stream; no tensor is
allocated, transferred or retained by the native function. Graphs/overlap and
multi-request operation are outside the CP executor's validated configuration.
"""
import ctypes
import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import torch


@functools.lru_cache(maxsize=2)
def _load_kernel(u16=False):
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("Fused IEEE FP32 CP affine is validated on SM100 only")
    from flashinfer.jit.env import CUTLASS_INCLUDE_DIRS
    source = Path(__file__).with_name("affine_simt.cu")
    includes = [Path(p) for p in CUTLASS_INCLUDE_DIRS]
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(compiler).is_file():
        raise RuntimeError("Fused FP32 CP affine requires the CUDA nvcc compiler")
    fingerprint = hashlib.sha256(source.read_bytes())
    fingerprint.update(source.with_name("paired_mma.cuh").read_bytes())
    for name in ("cutlass/gemm/device/gemm_batched.h", "cutlass/gemm/kernel/default_gemm.h",
                 "cutlass/gemm/threadblock/default_mma_core_simt.h",
                 "cutlass/epilogue/thread/linear_combination.h",
                 "cutlass/gemm/threadblock/mma_pipelined.h",
                 "cutlass/transform/threadblock/predicated_tile_iterator.h"):
        fingerprint.update((includes[0] / name).read_bytes())
    fingerprint.update(json.dumps((torch.__version__, torch.version.cuda,
                                  platform.machine(), platform.libc_ver())).encode())
    key = fingerprint.hexdigest()
    cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "sglang/qwen35_cp_fp32"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / (key + ".so")
    with (cache / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            temporary = cache / (key + ".tmp-%d.so" % os.getpid())
            command = [compiler, str(source), "-shared", "-Xcompiler", "-fPIC", "-O3",
                       "-std=c++17", "-arch=sm_100", "-o", str(temporary)]
            command += ["-I" + str(p) for p in includes]
            started = time.monotonic()
            log = cache / (key + ".compile.log")
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError("FP32 CP affine compilation failed; see " + str(log))
            os.replace(str(temporary), str(library))
            (cache / (key + ".json")).write_text(json.dumps(dict(
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), key=key,
                command=command, seconds=time.monotonic()-started), indent=2) + "\n")
    handle = ctypes.CDLL(str(library))
    function = handle.qwen35_fp32_affine_u16 if u16 else handle.qwen35_fp32_affine
    function.argtypes = [ctypes.c_void_p] * (4 if u16 else 3) + [ctypes.c_int, ctypes.c_void_p]
    function.restype = ctypes.c_int
    # Retain the library handle with the callable for the worker's lifetime.
    return handle, function, library


def _launch(output, u32, state):
    with torch.cuda.device(output.device):
        u16 = u32.dtype == torch.bfloat16
        _, function, _ = _load_kernel(u16)
        args = [output.data_ptr(), u32.data_ptr(), state.data_ptr()]
        if u16:
            from .zero_mask import prepare

            args.append(prepare(output, u32, state).data_ptr())
        status = function(*args, len(u32), torch.cuda.current_stream(output.device).cuda_stream)
    if status:
        raise RuntimeError("CUTLASS FP32 CP affine returned status %d" % status)
    return output
