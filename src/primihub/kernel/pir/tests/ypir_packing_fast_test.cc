/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_packing_fast_test — condense/uncondense round-trip + lazy Barrett
 * congruence. Pure, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing_fast.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
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

TEST(YpirPackingFastTest, FastMultiplyNoReduce_Reduce_MatchesMultiplyNtt) {
  const auto p = P(8);
  const std::size_t pl = 8, blk = 2 * pl, K = 4;
  const std::uint64_t q[2] = {kQ0, kQ1};
  std::mt19937_64 rng(31);
  PolyMatrixNTT a, b;  // a: 1xK, b: Kx1, uncondensed
  a.rows = 1; a.cols = K; a.data.resize(K * blk);
  b.rows = K; b.cols = 1; b.data.resize(K * blk);
  for (std::size_t k = 0; k < K; ++k)
    for (std::size_t m = 0; m < 2; ++m)
      for (std::size_t z = 0; z < pl; ++z) {
        a.Poly(0, k, blk)[m * pl + z] = rng() % q[m];
        b.Poly(k, 0, blk)[m * pl + z] = rng() % q[m];
      }
  const auto ref = MultiplyNtt(p, a, b);                  // 1x1, reduced
  auto fast = FastMultiplyNoReduce(p, CondenseMatrix(p, a), CondenseMatrix(p, b));
  FastReduce(p, fast);
  EXPECT_EQ(fast.data, ref.data);
}

TEST(YpirPackingFastTest, FastAddNoReduce_Reduce_MatchesAdd) {
  const auto p = P(8);
  const std::size_t blk = 2 * 8;
  const std::uint64_t q[2] = {kQ0, kQ1};
  std::mt19937_64 rng(41);
  PolyMatrixNTT a, b;
  a.rows = 1; a.cols = 1; a.data.resize(blk);
  b.rows = 1; b.cols = 1; b.data.resize(blk);
  for (std::size_t m = 0; m < 2; ++m)
    for (std::size_t z = 0; z < 8; ++z) {
      a.data[m * 8 + z] = rng() % q[m];
      b.data[m * 8 + z] = rng() % q[m];
    }
  const auto ref = AddNtt(p, a, b);
  PolyMatrixNTT acc = a;
  FastAddIntoNoReduce(acc, b);
  FastReduce(p, acc);
  EXPECT_EQ(acc.data, ref.data);
}

TEST(YpirPackingFastTest, ProduceTable_MatchesUpstream) {
  EXPECT_EQ(ProduceTable(8, 2),
            (std::vector<std::size_t>{1, 0, 3, 2, 5, 4, 7, 6}));
  EXPECT_EQ(ProduceTable(8, 4),
            (std::vector<std::size_t>{2, 3, 1, 0, 4, 5, 6, 7}));
  EXPECT_EQ(ProduceTable(8, 8),
            (std::vector<std::size_t>{1, 0, 2, 3, 5, 4, 6, 7}));
  EXPECT_EQ(ProduceTable(16, 4),
            (std::vector<std::size_t>{2, 3, 1, 0, 6, 7, 5, 4, 8, 9, 10, 11,
                                      12, 13, 14, 15}));
  std::vector<std::size_t> t = ProduceTable(16, 4);
  std::sort(t.begin(), t.end());
  for (std::size_t i = 0; i < t.size(); ++i) EXPECT_EQ(t[i], i);
}

TEST(YpirPackingFastTest, AutomorphNttTables_PerLevel) {
  const std::vector<std::vector<std::size_t>> tabs = AutomorphNttTables(8, 3);
  ASSERT_EQ(tabs.size(), 3u);
  EXPECT_EQ(tabs[0], (std::vector<std::size_t>{1, 0, 3, 2, 5, 4, 7, 6}));
  EXPECT_EQ(tabs[1], (std::vector<std::size_t>{2, 3, 1, 0, 4, 5, 6, 7}));
  EXPECT_EQ(tabs[2], (std::vector<std::size_t>{1, 0, 2, 3, 5, 4, 6, 7}));
}

}  // namespace
}  // namespace primihub::pir::ypir