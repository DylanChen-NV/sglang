// IEEE FP32 SIMT correction with a separate FP32 add and final BF16 store.
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/epilogue/thread/linear_combination.h"

extern "C" int qwen35_fp32_affine(void* output, const void* u, const void* state,
                                int rows, void* stream) {
  using BF = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::GemmBatched<
      float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor,
      BF, cutlass::layout::RowMajor, float,
      cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 128, 8>,
      cutlass::gemm::GemmShape<32, 64, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<BF, 1, float, float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      2, 1, 1, cutlass::arch::OpMultiplyAdd>;
  // Head batches are interleaved in [token, head, value]. S is [head,V,K].
  // Accumulators start at zero; alpha=beta=1 is applied only in the epilogue.
  Gemm::Arguments args(
      {rows, 128, 128},
      {static_cast<const float*>(u), 4096}, 128,
      {static_cast<const float*>(state), 128}, 16384,
      {static_cast<const BF*>(output), 4096}, 128,
      {static_cast<BF*>(output), 4096}, 128,
      {1.0f, 1.0f}, 32);
  auto status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) return static_cast<int>(status);
  if (Gemm::get_workspace_size(args)) return -2;
  Gemm operation;
  return static_cast<int>(operation(args, nullptr, static_cast<cudaStream_t>(stream)));
}
