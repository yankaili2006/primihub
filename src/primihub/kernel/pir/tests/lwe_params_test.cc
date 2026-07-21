/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * LweParams tests. Pure arithmetic, no kernel dependency — runs in
 * both stub and vendored modes.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"

namespace primihub::pir::core {
namespace {

TEST(LweParamsTest, EmbeddedTableHasAllRowsFromUpstream) {
  // Upstream params.csv at commit e9020b03 has 9 data rows.
  EXPECT_EQ(kLweParamEntryCount, 9u);
  for (std::size_t i = 0; i < kLweParamEntryCount; ++i) {
    const auto& row = kLweParamEntries[i];
    // Invariants the upstream CSV maintains.
    EXPECT_EQ(row.log_n, 10u) << "row " << i;
    EXPECT_EQ(row.log_q, 32u) << "row " << i;
    EXPECT_DOUBLE_EQ(row.sigma, 6.4) << "row " << i;
    EXPECT_GT(row.p_simple, 0u) << "row " << i;
    EXPECT_GT(row.p_double, 0u) << "row " << i;
    EXPECT_LT(row.p_double, row.p_simple) << "row " << i
        << ": DoublePIR's plaintext modulus is always smaller than "
           "SimplePIR's (it carries one extra layer of LWE noise).";
  }
}

TEST(LweParamsTest, PickFailsWhenNotInitialized) {
  LweParams p;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/false, /*samples=*/1 << 13, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("n=0"), std::string::npos)
      << "must guide caller to set n+logq; got: " << err;
}

TEST(LweParamsTest, PickSimplePirSmallestRow) {
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/false, /*samples=*/1 << 13, &err);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  EXPECT_DOUBLE_EQ(p.sigma, 6.4);
  EXPECT_EQ(p.p, 991u) << "log_m=13 row p_simple is 991 per upstream csv";
}

TEST(LweParamsTest, PickDoublePirSmallestRow) {
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/true, /*samples=*/1 << 13, &err);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  EXPECT_DOUBLE_EQ(p.sigma, 6.4);
  EXPECT_EQ(p.p, 929u) << "log_m=13 row p_double is 929 per upstream csv";
}

TEST(LweParamsTest, PickClimbsToLargerRowWhenSamplesExceedSmallest) {
  // samples = 2^17 + 1 — first row that fits is log_m = 18.
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/false,
                   /*samples=*/(static_cast<uint64_t>(1) << 17) + 1, &err);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  EXPECT_EQ(p.p, 416u) << "log_m=18 row p_simple is 416";
}

TEST(LweParamsTest, PickFailsWhenNoRowMatches) {
  LweParams p;
  p.n = 2048;  // not in the table — only n=1024 rows exist
  p.logq = 32;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/false, /*samples=*/1 << 13, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("no row matches"), std::string::npos)
      << "must include the search criteria for diagnosability; got: "
      << err;
  EXPECT_NE(err.find("@simplepir WORKSPACE_GITHUB pin"), std::string::npos)
      << "must point at the upstream pin so contributors know how to "
         "extend the table; got: "
      << err;
}

TEST(LweParamsTest, PickFailsWhenSamplesExceedAllRows) {
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  auto rc = p.Pick(/*doublepir=*/false,
                   /*samples=*/(static_cast<uint64_t>(1) << 22), &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("no row matches"), std::string::npos);
}

TEST(LweParamsTest, DeltaMatchesFormula) {
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  ASSERT_EQ(p.Pick(/*doublepir=*/false, /*samples=*/1 << 13, &err),
            retcode::SUCCESS) << err;
  // p_simple = 991 at log_m=13. Delta = 2^32 / 991.
  const uint64_t expected = (static_cast<uint64_t>(1) << 32) / 991ULL;
  EXPECT_EQ(p.Delta(), expected);
}

TEST(LweParamsTest, RoundCenterMapsToBoundary) {
  LweParams p;
  p.n = 1024;
  p.logq = 32;
  std::string err;
  ASSERT_EQ(p.Pick(/*doublepir=*/false, /*samples=*/1 << 13, &err),
            retcode::SUCCESS) << err;
  const uint64_t d = p.Delta();
  // Round(0) = 0.
  EXPECT_EQ(p.Round(0), 0u);
  // Round(Delta) = 1 mod p.
  EXPECT_EQ(p.Round(d), 1u % p.p);
  // Round(Delta * 5 + Delta/3) — within the rounding bucket of 5.
  EXPECT_EQ(p.Round(d * 5 + d / 3), 5u % p.p);
  // Round(Delta * 5 - Delta/3) — also within bucket 5 thanks to the
  // (x + Delta/2) trick.
  EXPECT_EQ(p.Round(d * 5 - d / 3), 5u % p.p);
}


// ---- NumBasePDigits (chunk 4a of DoublePIR port; task 5.5 dep) ----

TEST(LweParamsTest, NumBasePDigitsPowerOfTwoP) {
  // p = 4, logq = 32 -> ceil(32 / 2) = 16 digits.
  LweParams params;
  params.logq = 32;
  params.p = 4;
  EXPECT_EQ(params.NumBasePDigits(), 16u);
}

TEST(LweParamsTest, NumBasePDigitsArbitraryP) {
  // p = 929 (from upstream params CSV log_n=10 log_m=13), logq = 32.
  // log2(929) ~= 9.860; ceil(32 / 9.860) = ceil(3.245) = 4 digits.
  LweParams params;
  params.logq = 32;
  params.p = 929;
  EXPECT_EQ(params.NumBasePDigits(), 4u);

  // p = 781 (log_n=10 log_m=14): log2(781) ~= 9.609; ceil(32 / 9.609) = 4.
  params.p = 781;
  EXPECT_EQ(params.NumBasePDigits(), 4u);
}

TEST(LweParamsTest, NumBasePDigitsLargePForcesOneDigit) {
  // When p > 2^logq, ceil(32 / 33) = 1.
  LweParams params;
  params.logq = 32;
  params.p = 1ULL << 33;
  EXPECT_EQ(params.NumBasePDigits(), 1u);
}

TEST(LweParamsDeathTest, NumBasePDigitsFatalOnZeroP) {
  LweParams params;
  params.logq = 32;
  params.p = 0;
  EXPECT_DEATH(params.NumBasePDigits(), "NumBasePDigits");
}

TEST(LweParamsDeathTest, NumBasePDigitsFatalOnPOne) {
  LweParams params;
  params.logq = 32;
  params.p = 1;
  EXPECT_DEATH(params.NumBasePDigits(), "NumBasePDigits");
}

}  // namespace
}  // namespace primihub::pir::core
