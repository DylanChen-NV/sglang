# Qwen3.5 CP pre-rounding GDN output experiment

The five vendored files come from the verified portable FlashInfer 0.7.1
runtime used for the native TP2 and CP controls. Their original license headers
are retained. The exact upstream hashes are checked by `__init__.py` before use.
This experiment does not upgrade or rewrite the installed FlashInfer package.

The kernel adds a global FP32 side output immediately before the stock BF16
output conversion. The stock arithmetic, BF16 epilogue, state and checkpoint
paths remain for direct bitwise verification. Direct stores avoid enlarging
the shared-memory output buffer. Padded chunk rows are masked.

Both the non-CP kernel and the GPU-internal CP main prefill kernel expose the
side output. The wrapper preserves upstream's automatic parallelism heuristic,
and CP T/MN precompute and state fixup use their unchanged upstream kernels.
The adapters add the side-output argument and use independent kernel classes
or JIT module names. The scheduler is unmodified.
The public wrapper intentionally supports only the pinned SM100 BF16 Qwen3.5
CP shape and FP32 states. Local-output experiments keep the pre-rounding FP32
accumulator. The optional `u-output` FP32 affine optimization instead writes
`FP32(BF16(acc))` directly into caller-owned U32 scratch, preserving exactly
the upstream BF16 response values while avoiding a separate conversion.
The rounding policy belongs to both CP and non-CP compilation cache keys.
The BF16 epilogue is retained for bitwise checks; persistent caches and state
formats are unchanged. The two output policies must never share a compiled
specialization that omits the rounding flag.
