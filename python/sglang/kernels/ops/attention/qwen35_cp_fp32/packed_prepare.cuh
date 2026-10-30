// Exact-bit preparation for the U16 IEEE correction.
#pragma once
#include <cstdint>

namespace qwen35_cp {
__device__ __forceinline__ unsigned block_or(unsigned value, unsigned *shared) {
  unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  unsigned bits = __reduce_or_sync(0xffffffffu, value);
  if (lane == 0)
    shared[warp] = bits;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned all = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
      all |= shared[i];
    shared[0] = all;
  }
  __syncthreads();
  return shared[0];
}
__device__ __forceinline__ uint4 load_pack(std::uint16_t const *ptr,
                                           bool aligned) {
  if (aligned)
    return *reinterpret_cast<uint4 const *>(ptr);
  uint4 value;
  unsigned *words = reinterpret_cast<unsigned *>(&value);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    words[i] = unsigned(ptr[2 * i]) | (unsigned(ptr[2 * i + 1]) << 16);
  return value;
}
__device__ __forceinline__ void store_pack(std::uint16_t *ptr, uint4 value,
                                           bool aligned) {
  if (aligned) {
    *reinterpret_cast<uint4 *>(ptr) = value;
    return;
  }
  unsigned const *words = reinterpret_cast<unsigned const *>(&value);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    ptr[2 * i] = std::uint16_t(words[i]);
    ptr[2 * i + 1] = std::uint16_t(words[i] >> 16);
  }
}
__device__ __forceinline__ bool nonfinite(unsigned bits) {
  return (bits & 0x7fffffffu) >= 0x7f800000u;
}
__device__ __forceinline__ bool bf16_nan(unsigned bits) {
  return ((bits & 0x7fffu) > 0x7f80u) || (((bits >> 16) & 0x7fffu) > 0x7f80u);
}
__device__ __forceinline__ unsigned canonical(unsigned bits) {
  unsigned mask = ((bits & 0xffffu) == 0x8000u ? 0x8000u : 0u) |
                  ((bits >> 16) == 0x8000u ? 0x80000000u : 0u);
  return bits & ~mask;
}
// Separate CTAs check state heads and U tiles. GEMM consumes both flags after
// this kernel; no CTA waits for another CTA. Contiguous offset views are valid.
__global__ void packed_prepare(std::uint16_t *out, std::uint16_t const *u,
                               unsigned const *state, int *flags, int rows,
                               int tiles) {
  __shared__ unsigned reduction[8];
  int head = blockIdx.x / (tiles + 1), tile = blockIdx.x % (tiles + 1),
      tid = threadIdx.x;
  if (tile == tiles) {
    auto ptr = state + head * 16384;
    bool aligned = (reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0;
    unsigned bad = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      auto p = ptr + 4 * (tid + i * 256);
      uint4 v = aligned ? *reinterpret_cast<uint4 const *>(p)
                        : uint4{p[0], p[1], p[2], p[3]};
      bad |= nonfinite(v.x) | nonfinite(v.y) | nonfinite(v.z) | nonfinite(v.w);
    }
    bad = block_or(bad, reduction);
    if (tid == 0)
      flags[head * (tiles + 1) + tiles] = !bad;
    return;
  }
  bool ua = (reinterpret_cast<std::uintptr_t>(u) & 15u) == 0;
  bool oa = (reinterpret_cast<std::uintptr_t>(out) & 15u) == 0;
  unsigned nonzero = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tile * 64 + tid / 16 + i * 16, col = (tid % 16) * 8;
    uint4 v = {0, 0, 0, 0};
    if (row < rows)
      v = load_pack(u + (row * 32 + head) * 128 + col, ua);
    nonzero |= (v.x | v.y | v.z | v.w) & 0x7fff7fffu;
  }
  nonzero = block_or(nonzero, reduction);
  if (nonzero) {
    if (tid == 0)
      flags[head * (tiles + 1) + tile] = 1;
    return;
  }
  uint4 old[4];
  unsigned nan = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int row = tile * 64 + tid / 16 + i * 16, col = (tid % 16) * 8;
    old[i] = {0, 0, 0, 0};
    if (row < rows)
      old[i] = load_pack(out + (row * 32 + head) * 128 + col, oa);
    nan |= bf16_nan(old[i].x) | bf16_nan(old[i].y) | bf16_nan(old[i].z) |
           bf16_nan(old[i].w);
  }
  nan = block_or(nan, reduction);
  if (!nan) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint4 v = {canonical(old[i].x), canonical(old[i].y), canonical(old[i].z),
                 canonical(old[i].w)};
      if (v.x != old[i].x || v.y != old[i].y || v.z != old[i].z ||
          v.w != old[i].w) {
        int row = tile * 64 + tid / 16 + i * 16, col = (tid % 16) * 8;
        if (row < rows)
          store_pack(out + (row * 32 + head) * 128 + col, v, oa);
      }
    }
  }
  if (tid == 0)
    flags[head * (tiles + 1) + tile] = !!nan;
}
} // namespace qwen35_cp
