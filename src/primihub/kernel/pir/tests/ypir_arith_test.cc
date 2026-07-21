/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_arith_test — P0. Verifies the Barrett port against direct modulo
 * and the exact constants/vectors from spiral-rs src/arith.rs tests.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

#include <cstdint>
#include <random>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kMod = 66974689739603969ull;
constexpr std::uint64_t kCr0 = 7906011006380390721ull;
constexpr std::uint64_t kCr1 = 275ull;

TEST(YpirArithTest, GetBarrettCrs_MatchesUpstreamConstants) {
  const auto crs = GetBarrettCrs(kMod);
  EXPECT_EQ(crs.first, kCr0);   // cr0 = low 64 of floor(2^128/q)
  EXPECT_EQ(crs.second, kCr1);  // cr1 = high 64
}

TEST(YpirArithTest, BarrettReductionU128_HandComputed) {
  const __uint128_t m = kMod;
  EXPECT_EQ(BarrettReductionU128Raw(kMod, kCr0, kCr1, m), 0u);
  EXPECT_EQ(BarrettReductionU128Raw(kMod, kCr0, kCr1, m + 1), 1u);
  EXPECT_EQ(BarrettReductionU128Raw(kMod, kCr0, kCr1, m * 7 + 5), 5u);
}

TEST(YpirArithTest, BarrettReductionU128_MatchesModulo_Sweep) {
  std::mt19937_64 rng(0xBA12E77u);
  for (int i = 0; i < 5000; ++i) {
    const __uint128_t val =
        (static_cast<__uint128_t>(rng()) << 64) | rng();
    EXPECT_EQ(BarrettReductionU128Raw(kMod, kCr0, kCr1, val),
              static_cast<std::uint64_t>(val % kMod));
  }
}

TEST(YpirArithTest, BarrettRawU64_MatchesModulo_Sweep) {
  std::mt19937_64 rng(0x5EED1234u);
  for (int i = 0; i < 5000; ++i) {
    const std::uint64_t val = rng();
    EXPECT_EQ(BarrettRawU64(val, kCr1, kMod), val % kMod);
  }
}

TEST(YpirArithTest, DefaultModuli_BarrettMatchesModulo) {
  // The two ypir CRT moduli (DEFAULT_MODULI), ~2^28.
  for (std::uint64_t q : {268369921ull, 249561089ull}) {
    const auto crs = GetBarrettCrs(q);
    std::mt19937_64 rng(q);
    for (int i = 0; i < 2000; ++i) {
      const std::uint64_t a = rng() % q;
      const std::uint64_t b = rng() % q;
      EXPECT_EQ(MultiplyModular(a, b, q, crs.first, crs.second, 2),
                (a * b) % q)
          << "q=" << q << " a=" << a << " b=" << b;
    }
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
