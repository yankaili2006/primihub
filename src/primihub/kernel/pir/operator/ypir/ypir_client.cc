/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

namespace primihub::pir::ypir {

std::vector<std::uint32_t> GenerateMatrixRing(ChaChaRng& rng_pub, std::size_t n,
                                              std::size_t rows,
                                              std::size_t cols) {
  assert(rows % n == 0);
  assert(cols % n == 0);
  const std::size_t rows_outer = rows / n;
  const std::size_t cols_outer = cols / n;

  std::vector<std::uint32_t> out(rows * cols, 0);
  for (std::size_t i = 0; i < rows_outer; ++i) {
    for (std::size_t j = 0; j < cols_outer; ++j) {
      std::vector<std::uint32_t> a(n);
      for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng_pub.NextU32();
      const std::vector<std::uint32_t> mat = NegacyclicMatrixU32(a);
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          out[(i * n + k) * cols + (j * n + l)] = mat[k * n + l];
    }
  }
  return out;
}

}  // namespace primihub::pir::ypir
