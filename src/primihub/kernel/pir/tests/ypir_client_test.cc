/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_client (chunk 12): GenerateMatrixRing. Verifies the
 * block-negacyclic layout + ChaCha20 sampling order against an independent
 * re-derivation (same seed replayed) and an explicit per-block placement
 * check using NegacyclicMatrixU32. Pure scalar logic, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

namespace primihub::pir::ypir {
namespace {

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirClientTest, GenerateMatrixRing_BlockNegacyclicLayout) {
  const std::size_t n = 2, rows = 4, cols = 6;  // rows_outer=2, cols_outer=3
  ChaChaRng rng = ChaChaRng::FromSeed(Seed(7));
  const std::vector<std::uint32_t> out = GenerateMatrixRing(rng, n, rows, cols);
  ASSERT_EQ(out.size(), rows * cols);

  // Independent re-derivation: replay the same stream and place each n x n
  // negacyclic block, matching upstream's (i outer, j inner) sampling order.
  ChaChaRng rng2 = ChaChaRng::FromSeed(Seed(7));
  std::vector<std::uint32_t> expected(rows * cols, 0);
  const std::size_t rows_outer = rows / n, cols_outer = cols / n;
  for (std::size_t i = 0; i < rows_outer; ++i) {
    for (std::size_t j = 0; j < cols_outer; ++j) {
      std::vector<std::uint32_t> a(n);
      for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng2.NextU32();
      const std::vector<std::uint32_t> mat = NegacyclicMatrixU32(a);
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          expected[(i * n + k) * cols + (j * n + l)] = mat[k * n + l];
    }
  }
  EXPECT_EQ(out, expected);
}

// Larger n (4) and a single block to exercise a different shape.
TEST(YpirClientTest, GenerateMatrixRing_SingleBlockN4) {
  const std::size_t n = 4, rows = 4, cols = 4;  // one block
  ChaChaRng rng = ChaChaRng::FromSeed(Seed(19));
  const std::vector<std::uint32_t> out = GenerateMatrixRing(rng, n, rows, cols);

  ChaChaRng rng2 = ChaChaRng::FromSeed(Seed(19));
  std::vector<std::uint32_t> a(n);
  for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng2.NextU32();
  const std::vector<std::uint32_t> mat = NegacyclicMatrixU32(a);
  // single block: out == mat exactly (row-major n x n into cols=n)
  ASSERT_EQ(out.size(), mat.size());
  EXPECT_EQ(out, mat);
}

}  // namespace
}  // namespace primihub::pir::ypir
