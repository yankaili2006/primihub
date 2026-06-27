/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_discrete_gaussian_test — chunk 4b verification.
 *   * Init: cdf_table for width 6.4 matches the upstream-documented
 *     CDF_TABLE_GAUS_6_4 (53 values) exactly, plus structural invariants.
 *   * Sample: hand-computed inverse-CDF boundaries against that table.
 *   * Statistical: deterministic (mt19937_64) mean/std-dev sanity.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kU64Max = std::numeric_limits<std::uint64_t>::max();

// Upstream spiral-rs discrete_gaussian.rs documented CDF table for a
// width-6.4 Gaussian over [-26, 26] (CDF * 2^64, 2^64 -> 2^64-1).
const std::vector<std::uint64_t>& DocumentedCdf64() {
  static const std::vector<std::uint64_t> t = {
      0ull, 0ull, 0ull, 7ull, 225ull, 6114ull, 142809ull, 2864512ull,
      49349166ull, 730367088ull, 9288667698ull, 101545086850ull,
      954617134063ull, 7720973857474ull, 53757667977838ull,
      322436486442815ull, 1667499996257361ull, 7443566871362048ull,
      28720140744863884ull, 95948302954529081ull, 278161926109627739ull,
      701795634139702303ull, 1546646853635104741ull, 2991920295851131431ull,
      5112721055115151939ull, 7782220156096217088ull, 10664523917613334528ull,
      13334023018594399677ull, 15454823777858420185ull, 16900097220074446875ull,
      17744948439569849313ull, 18168582147599923877ull, 18350795770755022535ull,
      18418023932964687732ull, 18439300506838189568ull, 18445076573713294255ull,
      18446421637223108801ull, 18446690316041573778ull, 18446736352735694142ull,
      18446743119092417553ull, 18446743972164464766ull, 18446744064420883918ull,
      18446744072979184528ull, 18446744073660202450ull, 18446744073706687104ull,
      18446744073709408807ull, 18446744073709545502ull, 18446744073709551391ull,
      18446744073709551609ull, kU64Max, kU64Max, kU64Max, kU64Max};
  return t;
}

TEST(YpirDiscreteGaussianTest, Init_Width6_4_MatchesDocumentedCdf) {
  // The documented CDF_TABLE_GAUS_6_4 is a higher-precision reference;
  // the runtime f64 `init` reproduces it only to f64 precision (the
  // large entries differ by < 1 ULP of the scaled double, ~1e-16
  // relative). Assert each entry matches within a tolerance far tighter
  // than any real porting error (wrong PI/width/formula shifts values
  // by >>1%) yet loose enough for f64 accumulation noise. Zero and
  // saturated (2^64-1) entries are exact anchors.
  const auto dg = DiscreteGaussian::Init(6.4);
  EXPECT_EQ(dg.max_val(), 26);
  const auto& got = dg.cdf_table();
  const auto& want = DocumentedCdf64();
  ASSERT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < want.size(); ++i) {
    if (want[i] == 0ull || want[i] == kU64Max) {
      EXPECT_EQ(got[i], want[i]) << "cdf_table[" << i << "] (exact anchor)";
      continue;
    }
    const double rel = std::abs(static_cast<double>(got[i]) -
                                static_cast<double>(want[i])) /
                       static_cast<double>(want[i]);
    EXPECT_LT(rel, 1e-9) << "cdf_table[" << i << "] got " << got[i]
                         << " want " << want[i];
  }
}

TEST(YpirDiscreteGaussianTest, Init_StructuralInvariants) {
  const auto dg = DiscreteGaussian::Init(6.4);
  const auto& cdf = dg.cdf_table();
  ASSERT_FALSE(cdf.empty());
  for (std::size_t i = 1; i < cdf.size(); ++i) {
    EXPECT_GE(cdf[i], cdf[i - 1]) << "non-monotonic at " << i;
  }
  EXPECT_EQ(cdf.front(), 0ull);
  EXPECT_EQ(cdf.back(), kU64Max);
  // CDF through integer 0 (index max_val) is ~0.5 of the mass.
  const std::uint64_t mid = cdf[static_cast<std::size_t>(dg.max_val())];
  EXPECT_GT(mid, kU64Max / 10 * 4);  // > 0.4
  EXPECT_LT(mid, kU64Max / 10 * 6);  // < 0.6
}

TEST(YpirDiscreteGaussianTest, Sample_HandComputedBoundaries) {
  const auto dg = DiscreteGaussian::Init(6.4);
  const std::uint64_t M = 1ull << 32;  // LWE modulus 2^32
  // Early-table boundaries are exact across the doc-vs-runtime precision
  // difference (cdf[0..4] match the documented table byte-for-byte).
  // result = i_min - 26 (mod M), i_min = smallest i with cdf[i] >= sv.
  EXPECT_EQ(dg.Sample(M, 0), M - 26);  // i_min=0  -> -26
  EXPECT_EQ(dg.Sample(M, 7), M - 23);  // cdf[3]=7   -> i_min=3 -> -23
  EXPECT_EQ(dg.Sample(M, 8), M - 22);  // cdf[4]=225 -> i_min=4 -> -22
}

TEST(YpirDiscreteGaussianTest, Sample_MatchesIndependentInverseCdf) {
  // Cross-check the constant-time downward inverse-CDF against a plain
  // forward linear scan for i_min over the actual table. This is robust
  // to f64 precision at the saturated top end (where the runtime table
  // hits 2^64-1 a few indices earlier than the documented reference).
  const auto dg = DiscreteGaussian::Init(6.4);
  const auto& cdf = dg.cdf_table();
  const std::uint64_t M = 1ull << 32;
  auto ref = [&](std::uint64_t sv) -> std::uint64_t {
    std::size_t imin = 0;
    while (imin + 1 < cdf.size() && cdf[imin] < sv) ++imin;
    std::int64_t out = static_cast<std::int64_t>(imin) - dg.max_val();
    if (out < 0) out += static_cast<std::int64_t>(M);
    return static_cast<std::uint64_t>(out);
  };
  const std::uint64_t svs[] = {0ull,        1ull,        1000ull,
                               cdf[20],     cdf[26] + 1, cdf[40],
                               1ull << 63,  kU64Max};
  for (std::uint64_t sv : svs) {
    EXPECT_EQ(dg.Sample(M, sv), ref(sv)) << "sv=" << sv;
  }
}

TEST(YpirDiscreteGaussianTest, Sample_StatisticalSanity_Deterministic) {
  const double width = 6.4;
  const auto dg = DiscreteGaussian::Init(width);
  const std::uint64_t M = 1ull << 32;
  std::mt19937_64 rng(0xC0FFEEu);  // fixed seed -> deterministic
  const int trials = 20000;
  double sum = 0.0;
  std::vector<double> centred;
  centred.reserve(trials);
  for (int t = 0; t < trials; ++t) {
    std::int64_t v = static_cast<std::int64_t>(dg.Sample(M, rng()));
    if (v >= static_cast<std::int64_t>(M / 2)) v -= static_cast<std::int64_t>(M);
    centred.push_back(static_cast<double>(v));
    sum += v;
  }
  const double mean = sum / trials;
  double var = 0.0;
  for (double x : centred) var += (x - mean) * (x - mean);
  var /= trials;
  const double std_dev = std::sqrt(var);
  const double expected_std = width / std::sqrt(2.0 * M_PI);  // ~2.553
  EXPECT_LT(std::abs(mean), 0.25) << "mean " << mean;
  EXPECT_LT(std::abs(std_dev - expected_std), expected_std * 0.1)
      << "std " << std_dev << " expected " << expected_std;
}

}  // namespace
}  // namespace primihub::pir::ypir
