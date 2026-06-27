// Microbench: breakdown of FrodoPIR Setup hot-path components.
#include <chrono>
#include <cstdio>
#include <vector>
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_flat_matrix.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

using namespace primihub::pir::frodo;
using clk = std::chrono::high_resolution_clock;

static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

int main() {
  const std::size_t dim = 512;
  const std::size_t m = 1000000;
  SeedBytes seed{};
  for (std::size_t i = 0; i < 32; ++i) seed[i] = std::uint8_t(i + 1);
  printf("FrodoPIR Setup component breakdown @ dim=%zu m=%zu (~%.2f GB)\n\n",
         dim, m, double(dim)*m*4.0/1e9);

  printf("== chunk g-5 flat path ==\n");
  auto t0 = clk::now();
  ColMajorMatrix flat_seeded = GenerateLweMatrixFromSeedFlat(seed, dim, m);
  printf("  GenerateLweMatrixFromSeedFlat:  %.1f ms\n", ms_since(t0));

  t0 = clk::now();
  ColMajorMatrix flat_swap = SwapMatrixFmtFlat(flat_seeded);
  printf("  SwapMatrixFmtFlat:              %.1f ms\n", ms_since(t0));
  printf("  flat output: height=%zu width=%zu\n\n",
         flat_swap.height(), flat_swap.width());

  printf("== prior nested path ==\n");
  t0 = clk::now();
  auto nested_seeded = GenerateLweMatrixFromSeed(seed, dim, m);
  printf("  GenerateLweMatrixFromSeed:      %.1f ms\n", ms_since(t0));

  t0 = clk::now();
  auto nested_swap = SwapMatrixFmt(nested_seeded);
  printf("  SwapMatrixFmt:                  %.1f ms\n", ms_since(t0));
  printf("  nested output: rows=%zu cols=%zu\n",
         nested_swap.size(),
         nested_swap.empty() ? 0u : nested_swap[0].size());
  return 0;
}
