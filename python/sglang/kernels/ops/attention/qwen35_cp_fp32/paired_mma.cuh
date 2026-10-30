// Paired IEEE FP32 arithmetic over the existing CUTLASS SIMT fragment layout.
#pragma once

namespace qwen35_cp {

__device__ __forceinline__ void pair_fma(
    float a, float b0, float b1, float& c0, float& c1) {
  float x, y;
  asm("{ .reg .b64 a,b,c,d; mov.b64 a,{%2,%2}; mov.b64 b,{%3,%4}; "
      "mov.b64 c,{%5,%6}; fma.rn.f32x2 d,a,b,c; mov.b64 {%0,%1},d; }"
      : "=f"(x), "=f"(y)
      : "f"(a), "f"(b0), "f"(b1), "f"(c0), "f"(c1));
  c0 = x;
  c1 = y;
}

template <class Base>
struct PairedWarpMma : Base {
  using FragmentA = typename Base::FragmentA;
  using FragmentB = typename Base::FragmentB;
  using FragmentC = typename Base::FragmentC;
  using ThreadShape = typename Base::ThreadMma::Shape;
  using LayoutA = typename Base::ThreadLayoutA;
  using LayoutB = typename Base::ThreadLayoutB;
  using LayoutC = typename Base::LayoutC;

  CUTLASS_DEVICE
  void operator()(FragmentC& d, FragmentA a, FragmentB b,
                  FragmentC const& c, int = 0) const {
    constexpr int M = ThreadShape::kM;
    constexpr int N = ThreadShape::kN;
    constexpr int K = ThreadShape::kK;
    static_assert(N % 2 == 0, "Paired FP32 requires an even thread N tile");
    auto la = LayoutA::packed({M, K});
    auto lb = LayoutB::packed({K, N});
    auto lc = LayoutC::packed({M, N});
    d = c;
    // Each output retains exactly the original increasing-K FMA sequence.
    // Pairing independent N outputs changes instruction issue, not reduction order.
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < K; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < N; n += 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < M; ++i) {
          int m = (n % 4) ? M - 1 - i : i;
          pair_fma(a[la({m, k})], b[lb({k, n})], b[lb({k, n + 1})],
                   d[lc({m, n})], d[lc({m, n + 1})]);
        }
      }
    }
  }
};

}  // namespace qwen35_cp
