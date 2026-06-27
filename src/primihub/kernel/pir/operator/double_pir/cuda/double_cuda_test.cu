// Standalone GPU validation of the DoublePIR LWE matmul kernel (task 2.2):
// C = A*B mod 2^32 vs a CPU reference, including the matrix-vector Answer case.
#include "double_cuda_kernels.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace primihub::pir::doublepir::cuda;

static bool CheckMatMul(std::size_t rows, std::size_t inner, std::size_t cols,
                        const char* label) {
  auto rnd = [](std::uint64_t x) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x;
  };
  std::vector<std::uint32_t> a(rows * inner), b(inner * cols);
  std::uint64_t s = 0xABCDEF1u;
  for (auto& v : a) { s = rnd(s + 1); v = static_cast<std::uint32_t>(s); }
  for (auto& v : b) { s = rnd(s + 1); v = static_cast<std::uint32_t>(s); }

  std::vector<std::uint32_t> got(rows * cols, 0), ref(rows * cols, 0);
  LweMatMulMod2Pow32(got.data(), a.data(), b.data(), rows, inner, cols);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      std::uint32_t acc = 0;
      for (std::size_t k = 0; k < inner; ++k)
        acc += a[i * inner + k] * b[k * cols + j];  // wraps mod 2^32
      ref[i * cols + j] = acc;
    }
  std::size_t mism = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    if (got[i] != ref[i]) ++mism;
  std::printf("LWE matmul %s (%zux%zu * %zux%zu): %zu mismatches -> %s\n", label,
              rows, inner, inner, cols, mism, mism == 0 ? "PASS" : "FAIL");
  return mism == 0;
}

int main() {
  bool ok = true;
  ok &= CheckMatMul(200, 300, 5, "batched");        // small batched queries
  ok &= CheckMatMul(256, 256, 1, "answer-vector");  // the matrix-vector Answer
  ok &= CheckMatMul(17, 33, 9, "ragged");           // non-tile-aligned dims
  return ok ? 0 : 1;
}
