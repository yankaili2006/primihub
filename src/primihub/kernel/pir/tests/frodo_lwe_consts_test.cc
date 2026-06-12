/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_lwe_consts_test — verifies the LWE constant arithmetic
 * port matches upstream Rust's `pub mod lwe` byte-for-byte at the
 * typical operating points (1, 8, 16 plaintext bits) plus the
 * overflow boundary at 32 bits.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_lwe_consts.h"

#include <cstdint>

#include <gtest/gtest.h>

namespace primihub::pir::frodo {
namespace {

TEST(FrodoLweConstsTest, Modulus_Equals_2pow32) {
  // u32::MAX as u64 + 1 = 4294967296 = 2^32.
  EXPECT_EQ(kLweModulus, 4294967296ULL);
}

TEST(FrodoLweConstsTest, PlaintextSize_BoundaryValues) {
  // 1 bit  -> 2
  // 8 bits -> 256
  // 16 bits -> 65536
  // 31 bits -> 2^31 = 2147483648
  // 32 bits -> 0 (overflow saturation per port comment)
  EXPECT_EQ(GetPlaintextSize(1), 2u);
  EXPECT_EQ(GetPlaintextSize(8), 256u);
  EXPECT_EQ(GetPlaintextSize(16), 65536u);
  EXPECT_EQ(GetPlaintextSize(31), 2147483648u);
  EXPECT_EQ(GetPlaintextSize(32), 0u)
      << "32+ plaintext_bits saturates to 0 per port contract; "
      << "upstream would panic at 2u32.pow(32) in debug mode.";
  EXPECT_EQ(GetPlaintextSize(64), 0u);
}

TEST(FrodoLweConstsTest, RoundingFactor_BoundaryValues) {
  // Rounding factor = 2^32 / plaintext_size.
  // 1 bit  -> 2^32 / 2 = 2^31 = 2147483648
  // 8 bits -> 2^32 / 256 = 2^24 = 16777216
  // 16 bits -> 2^32 / 65536 = 2^16 = 65536
  EXPECT_EQ(GetRoundingFactor(1), 2147483648u);
  EXPECT_EQ(GetRoundingFactor(8), 16777216u);
  EXPECT_EQ(GetRoundingFactor(16), 65536u);
}

TEST(FrodoLweConstsTest, RoundingFloor_IsHalfOfRoundingFactor) {
  EXPECT_EQ(GetRoundingFloor(1), GetRoundingFactor(1) / 2u);
  EXPECT_EQ(GetRoundingFloor(8), GetRoundingFactor(8) / 2u);
  EXPECT_EQ(GetRoundingFloor(16), GetRoundingFactor(16) / 2u);
  // Explicit numeric values:
  EXPECT_EQ(GetRoundingFloor(1), 1073741824u);   // 2^30
  EXPECT_EQ(GetRoundingFloor(8), 8388608u);      // 2^23
  EXPECT_EQ(GetRoundingFloor(16), 32768u);       // 2^15
}

TEST(FrodoLweConstsTest, OverflowBranch_RoundingFactor_ReturnsZero) {
  // When plaintext_size overflows to 0, rounding factor returns 0
  // rather than dividing by zero — verifies the guard in the port.
  EXPECT_EQ(GetRoundingFactor(32), 0u);
  EXPECT_EQ(GetRoundingFloor(32), 0u);
}

TEST(FrodoLweConstsTest, ConstexprEvaluability) {
  // The functions must be usable at compile time. constexpr in
  // template-arg position is the strongest assertion of this.
  constexpr std::uint32_t k_factor_8 = GetRoundingFactor(8);
  static_assert(k_factor_8 == 16777216u,
                "GetRoundingFactor(8) must be constexpr-evaluable "
                "and equal 2^24.");
  // Use the value to suppress unused-variable warnings.
  EXPECT_EQ(k_factor_8, 16777216u);
}

}  // namespace
}  // namespace primihub::pir::frodo
