/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_scheme (chunk 11): the seed machinery (StaticPublicSeed /
 * StaticSeed2 / GetSeed), YPIRParams, and the GetQPrime1/2 modulus helpers
 * (including the spiral-rs Q2_VALUES branch and the q2_bits==modulus_log2
 * full-modulus branch). Pure scalar logic, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"

#include <array>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

bool AllZeroFrom(const std::array<std::uint8_t, 32>& s, std::size_t from) {
  for (std::size_t i = from; i < 32; ++i)
    if (s[i] != 0) return false;
  return true;
}

TEST(YpirSchemeTest, StaticSeedsAndGetSeed) {
  const std::array<std::uint8_t, 32> pub = StaticPublicSeed();
  EXPECT_TRUE(AllZeroFrom(pub, 0));

  const std::array<std::uint8_t, 32> s2 = StaticSeed2();
  EXPECT_EQ(s2[0], 2);
  EXPECT_TRUE(AllZeroFrom(s2, 1));

  // get_seed(idx) = STATIC_PUBLIC_SEED with byte 0 = idx
  const std::array<std::uint8_t, 32> g0 = GetSeed(0);
  EXPECT_TRUE(AllZeroFrom(g0, 0));
  const std::array<std::uint8_t, 32> g1 = GetSeed(1);
  EXPECT_EQ(g1[0], 1);
  EXPECT_TRUE(AllZeroFrom(g1, 1));
  const std::array<std::uint8_t, 32> g5 = GetSeed(5);
  EXPECT_EQ(g5[0], 5);
  EXPECT_TRUE(AllZeroFrom(g5, 1));
}

TEST(YpirSchemeTest, YPIRParamsDefault) {
  YPIRParams yp;
  EXPECT_FALSE(yp.is_simplepir);
}

TEST(YpirSchemeTest, GetQPrime1IsFixed) {
  const Params p = Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                                1, 1, 1, 0, 0);
  EXPECT_EQ(GetQPrime1(p), static_cast<std::uint64_t>(1) << 20);  // 1048576
}

TEST(YpirSchemeTest, GetQPrime2Q2ValuesBranch) {
  Params p = Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, 1, 1,
                          1, 0, 0);
  ASSERT_NE(p.q2_bits, p.modulus_log2);  // product modulus is ~56 bits
  // Q2_VALUES[28] == 268369921
  EXPECT_EQ(GetQPrime2(p), static_cast<std::uint64_t>(268369921));
}

TEST(YpirSchemeTest, GetQPrime2FullModulusBranch) {
  Params p = Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, 1, 1,
                          1, 0, 0);
  // Force the equality branch: q2_bits == modulus_log2 -> full product modulus.
  p.q2_bits = p.modulus_log2;
  EXPECT_EQ(GetQPrime2(p), p.modulus);
}

}  // namespace
}  // namespace primihub::pir::ypir
