/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_packing_fast_test — condense/uncondense round-trip + lazy Barrett
 * congruence. Pure, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing_fast.h"

#include <cstdint>
#include <random>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

Params P(std::size_t pl) {
  return Params::Init(pl, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                      1, 1, 1, 0, 0);
}

TEST(YpirPackingFastTest, Condense_Uncondense_RoundTrips) {
  const auto p = P(8);
  const std::size_t blk = 2 * 8;
  PolyMatrixNTT a;
  a.rows = 2; a.cols = 2; a.data.resize(2 * 2 * blk);
  std::mt19937_64 rng(5);
  const std::uint64_t q[2] = {kQ0, kQ1};
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      for (std::size_t m = 0; m < 2; ++m)
        for (std::size_t z = 0; z < 8; ++z)
          a.Poly(i, j, blk)[m * 8 + z] = rng() % q[m];  // < 2^28 < 2^32
  EXPECT_EQ(UncondenseMatrix(p, CondenseMatrix(p, a)).data, a.data);
}

TEST(YpirPackingFastTest, FastBarrett_CongruentModulo) {
  const auto crs0 = GetBarrettCrs(kQ0);
  std::mt19937_64 rng(9);
  for (int i = 0; i < 5000; ++i) {
    const std::uint64_t v = rng();
    // result is congruent to v mod q (may be in [0, 2q)); reduce to compare.
    EXPECT_EQ(FastBarrettRawU64(v, crs0.second, kQ0) % kQ0, v % kQ0);
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
