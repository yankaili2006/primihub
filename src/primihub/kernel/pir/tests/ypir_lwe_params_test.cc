/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe_params_test — verifies the YPIR LWE parameter struct port
 * matches upstream Rust's `impl Default for LWEParams` exactly, and
 * ScaleK() reproduces upstream `scale_k()`.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirLweParamsTest, Default_MatchesUpstreamRust) {
  auto p = LweParams::Default();
  EXPECT_EQ(p.n, 1024u);
  // 2^32 = 4294967296. Verifies the `static_cast<uint64_t>(1) << 32`
  // didn't silently truncate to 32-bit (a common porting bug).
  EXPECT_EQ(p.modulus, static_cast<std::uint64_t>(4294967296ULL));
  EXPECT_EQ(p.pt_modulus, 256u);
  EXPECT_EQ(p.q2_bits, 28u);
  EXPECT_DOUBLE_EQ(p.noise_width, 27.57291103);
}

TEST(YpirLweParamsTest, DefaultNoiseWidth_MatchesMathDerivation) {
  // Upstream comment: `27.57291103, // 11 * sqrt(2*pi)`.
  // Verify the constant is within rounding tolerance of the math.
  const double expected = 11.0 * std::sqrt(2.0 * M_PI);
  // Upstream rounded to 8 decimals; allow a tight delta to catch
  // anyone replacing the literal with a different rounding.
  EXPECT_NEAR(kDefaultLweNoiseWidth, expected, 1e-6)
      << "kDefaultLweNoiseWidth (" << kDefaultLweNoiseWidth
      << ") drifted from upstream's 11*sqrt(2*pi) (" << expected
      << "). If upstream re-rounded, update both this test and the "
      << "header constant in the same commit.";
}

TEST(YpirLweParamsTest, ScaleK_Default_Equals_2pow24) {
  auto p = LweParams::Default();
  // 2^32 / 256 = 2^24 = 16777216.
  EXPECT_EQ(p.ScaleK(), static_cast<std::uint64_t>(16777216ULL));
}

TEST(YpirLweParamsTest, ScaleK_NonDivisible_TruncatesLikeRust) {
  // Rust `self.modulus / self.pt_modulus` is integer division —
  // truncates toward zero for u64. Verify C++ matches.
  LweParams p = LweParams::Default();
  p.modulus = 10;
  p.pt_modulus = 3;
  EXPECT_EQ(p.ScaleK(), 3u);  // 10 / 3 = 3 (rem 1)
}

TEST(YpirLweParamsTest, ScaleK_PtModulusGreaterThanModulus_ReturnsZero) {
  // Degenerate but well-defined: integer division 5 / 256 = 0.
  // The struct doesn't impose invariants — callers are responsible
  // for picking sane params. This test pins the lack of invariants
  // so a future "validate" method doesn't silently change behavior.
  LweParams p = LweParams::Default();
  p.modulus = 5;
  p.pt_modulus = 256;
  EXPECT_EQ(p.ScaleK(), 0u);
}

TEST(YpirLweParamsTest, StructIsTriviallyCopyable) {
  // The Rust struct derives Clone; the C++ port should remain
  // trivially copyable so that callers can pass it by value
  // without surprise. If a future field (e.g., a sampler ptr)
  // breaks this, the static_assert will flag it at compile time.
  // C++14-compatible form (no _v variable templates). The cc_test
  // toolchain on .50 does not default to C++17; using the legacy
  // `::value` form keeps the test portable.
  static_assert(std::is_trivially_copyable<LweParams>::value,
                "LweParams must stay trivially copyable so callers "
                "can pass it by value without surprise.");
  // Runtime no-op — the assertion above runs at compile time.
  SUCCEED();
}

}  // namespace
}  // namespace primihub::pir::ypir
