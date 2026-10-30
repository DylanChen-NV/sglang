// IEEE FP32 SIMT correction with a separate FP32 add and final BF16 store.
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "packed_prepare.cuh"
#include "paired_mma.cuh"
#include <cuda_runtime.h>

template <class Dense> struct Qwen35ZeroSkippingGemm {
  using SharedStorage = typename Dense::SharedStorage;
  static int const kThreadCount = Dense::kThreadCount;
  struct Params {
    typename Dense::Params gemm;
    int const *flags;
    int tiles;
  };
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &storage) {
    int const *flags = params.flags + blockIdx.z * (params.tiles + 1);
    if (flags[blockIdx.x] || !flags[params.tiles]) {
      Dense operation;
      operation(params.gemm, storage);
    }
  }
};

extern "C" int qwen35_fp32_affine(void *output, const void *u,
                                  const void *state, int rows, void *stream) {
  using BF = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::GemmBatched<
      float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, BF,
      cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 8>,
      cutlass::gemm::GemmShape<32, 64, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<BF, 1, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, 1,
      1, cutlass::arch::OpMultiplyAdd>;
  // Head batches are interleaved in [token, head, value]. S is [head,V,K].
  // Accumulators start at zero; alpha=beta=1 is applied only in the epilogue.
  Gemm::Arguments args({rows, 128, 128}, {static_cast<const float *>(u), 4096},
                       128, {static_cast<const float *>(state), 128}, 16384,
                       {static_cast<const BF *>(output), 4096}, 128,
                       {static_cast<BF *>(output), 4096}, 128, {1.0f, 1.0f},
                       32);
  auto status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess)
    return static_cast<int>(status);
  if (Gemm::get_workspace_size(args))
    return -2;
  Gemm operation;
  return static_cast<int>(
      operation(args, nullptr, static_cast<cudaStream_t>(stream)));
}

// Keep the original shared-memory iterators, FP32 reduction order and epilogue.
// Only global A loads use BF16; MmaPipelined's NumericArrayConverter promotes
// each loaded fragment exactly once, before its FP32 shared-memory store.
extern "C" int qwen35_fp32_affine_u16(void *output, const void *u,
                                      const void *state, int *flags, int rows,
                                      void *stream) {
  using BF = cutlass::bfloat16_t;
  using Reference = cutlass::gemm::device::GemmBatched<
      float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, BF,
      cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 8>,
      cutlass::gemm::GemmShape<32, 64, 8>, cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<BF, 1, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 2, 1,
      1, cutlass::arch::OpMultiplyAdd>;
  using Original = typename Reference::GemmKernel;
  using OldMma = typename Original::Mma;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<64, 8>, BF, cutlass::layout::RowMajor, 1,
      typename OldMma::IteratorA::ThreadMap, 1>;
  using OldPolicy = typename OldMma::Policy;
  using Policy = cutlass::gemm::threadblock::MmaPolicy<
      qwen35_cp::PairedWarpMma<typename OldPolicy::Operator>,
      typename OldPolicy::SmemPaddingA, typename OldPolicy::SmemPaddingB,
      OldPolicy::kPartitionsK>;
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      typename OldMma::Shape, IteratorA, typename OldMma::SmemIteratorA,
      typename OldMma::IteratorB, typename OldMma::SmemIteratorB, float,
      cutlass::layout::RowMajor, Policy>;
  using Swizzle = typename Reference::ThreadblockSwizzle;
  using Dense =
      cutlass::gemm::kernel::GemmBatched<Mma, typename Original::Epilogue,
                                         Swizzle>;
  using Kernel = Qwen35ZeroSkippingGemm<Dense>;
  if (rows <= 0)
    return -3;
  int tiles = (rows + 63) / 64;
  qwen35_cp::packed_prepare<<<32 * (tiles + 1), 256, 0,
                              static_cast<cudaStream_t>(stream)>>>(
      static_cast<std::uint16_t *>(output),
      static_cast<std::uint16_t const *>(u),
      static_cast<unsigned const *>(state), flags, rows, tiles);
  if (cudaGetLastError() != cudaSuccess)
    return -6;
  cutlass::gemm::GemmCoord problem(rows, 128, 128);
  Swizzle swizzle;
  auto tiled = swizzle.get_tiled_shape(problem, {64, 128, 8}, 32);
  typename Dense::Params dense{
      problem,
      tiled,
      {const_cast<BF *>(static_cast<const BF *>(u)), 4096},
      128,
      {const_cast<float *>(static_cast<const float *>(state)), 128},
      16384,
      {static_cast<BF *>(output), 4096},
      128,
      {static_cast<BF *>(output), 4096},
      128,
      {1.0f, 1.0f},
      32};
  typename Kernel::Params params{dense, flags, (rows + 63) / 64};
  int smem = int(sizeof(typename Kernel::SharedStorage));
  if (smem >= (48 << 10)) {
    auto error =
        cudaFuncSetAttribute(cutlass::Kernel<Kernel>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (error != cudaSuccess)
      return -4;
  }
  cutlass::Kernel<Kernel><<<swizzle.get_grid_shape(tiled), Kernel::kThreadCount,
                            smem, static_cast<cudaStream_t>(stream)>>>(params);
  return cudaGetLastError() == cudaSuccess ? 0 : -5;
}
