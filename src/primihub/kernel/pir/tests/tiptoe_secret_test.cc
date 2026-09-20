/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Unit test for the pure secret helpers (tiptoe chunk 1.1c). Zero-dependency,
 * runs in the default .50 build (no SEAL needed).
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_secret.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace primihub::pir::tiptoe {
namespace {

// rlwe BFV plaintext modulus (context_p) is 65537; exercise centering at it.
constexpr std::uint64_t kP = 65537;

TEST(TiptoeSecretTest, FromModuloPLowerHalfUnchanged) {
  // v <= p/2 maps to itself.
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, 0), 0);
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, 1), 1);
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, kP / 2), static_cast<std::int64_t>(kP / 2));
}

TEST(TiptoeSecretTest, FromModuloPUpperHalfCentersNegative) {
  // v > p/2 maps to v - p (negative representative).
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, kP - 1), -1);
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, kP - 2), -2);
  EXPECT_EQ(FromModuloP<std::int64_t>(kP, kP / 2 + 1),
            static_cast<std::int64_t>(kP / 2 + 1) - static_cast<std::int64_t>(kP));
}

TEST(TiptoeSecretTest, FromModuloPWrapsIntoUnsignedT) {
  // For an unsigned T, the negative representative wraps (mirrors Go T(realVal)).
  EXPECT_EQ(FromModuloP<std::uint32_t>(kP, kP - 1), 0xFFFFFFFFu);          // -1
  EXPECT_EQ(FromModuloP<std::uint32_t>(kP, kP - 2), 0xFFFFFFFEu);          // -2
  EXPECT_EQ(FromModuloP<std::uint32_t>(kP, 2), 2u);
}

TEST(TiptoeSecretTest, InRangeTernaryOnly) {
  EXPECT_TRUE(InRange<std::int64_t>(0));
  EXPECT_TRUE(InRange<std::int64_t>(1));
  EXPECT_TRUE(InRange<std::int64_t>(2));
  EXPECT_FALSE(InRange<std::int64_t>(3));
  EXPECT_FALSE(InRange<std::int64_t>(-1));
  EXPECT_EQ(kSecretMin, 0u);
  EXPECT_EQ(kSecretMax, 2u);
}

}  // namespace
}  // namespace primihub::pir::tiptoe
