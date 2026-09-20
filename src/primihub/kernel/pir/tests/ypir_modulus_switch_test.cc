/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_modulus_switch_test — chunk 4 verification.
 *   * Rescale: hand-computed centred-rounding cases.
 *   * Pack: exact byte layout for a single-coefficient fixture.
 *   * Roundtrip: clean plaintext slots survive pack -> recover exactly.
 *   * Soft boundaries: size mismatch / null out.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirModulusSwitchTest, Rescale_HandComputed) {
  // round(a * out / inp) over the centred residue, mirroring spiral.
  EXPECT_EQ(Rescale(0, 8, 4), 0u);
  EXPECT_EQ(Rescale(3, 8, 4), 2u);    // 3/8*4 = 1.5 -> 2
  EXPECT_EQ(Rescale(7, 8, 4), 3u);    // 7 centred = -1 -> -0.5 -> 3 (mod 4)
  EXPECT_EQ(Rescale(8, 64, 8), 1u);   // 8/64*8 = 1
  EXPECT_EQ(Rescale(40, 64, 8), 5u);  // 40 centred = -24 -> 5 (mod 8)
}

TEST(YpirModulusSwitchTest, Pack_ExactBytes_SingleCoeff) {
  // poly_len=1, modulus=64, q_1=q_2=8 -> 3 bits each, 6 bits -> 1 byte.
  // row0={8} -> Rescale(8,64,8)=1 at bit offset 0 (bit 0 set).
  // row1={16} -> Rescale(16,64,8)=2 at bit offset 3 (bit 4 set).
  const std::uint64_t row0[1] = {8};
  const std::uint64_t row1[1] = {16};
  const auto out = ModulusSwitchPack(row0, row1, /*poly_len=*/1,
                                     /*modulus=*/64, /*q_1=*/8, /*q_2=*/8);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], 0x11u);  // (1 << 0) | (1 << 4)
}

TEST(YpirModulusSwitchTest, Roundtrip_CleanValues_Exact) {
  // Multiples of modulus/q (= 8) are exact plaintext slots and survive
  // the lossy modulus switch without error, incl. values above mod/2.
  const std::uint64_t row0[4] = {0, 16, 32, 48};
  const std::uint64_t row1[4] = {8, 24, 40, 56};
  const std::uint64_t modulus = 64, q = 8;
  const auto ct = ModulusSwitchPack(row0, row1, 4, modulus, q, q);
  ASSERT_EQ(ct.size(), (3u + 3u) * 4u / 8u);  // 3 bytes

  std::vector<std::uint64_t> r0, r1;
  ASSERT_TRUE(ModulusSwitchRecover(ct, 4, modulus, q, q, &r0, &r1));
  ASSERT_EQ(r0.size(), 4u);
  ASSERT_EQ(r1.size(), 4u);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(r0[i], row0[i]) << "row0[" << i << "]";
    EXPECT_EQ(r1[i], row1[i]) << "row1[" << i << "]";
  }
}

TEST(YpirModulusSwitchTest, Recover_SizeMismatch_ReturnsFalse) {
  std::vector<std::uint8_t> wrong(2, 0);  // expected 3 bytes
  std::vector<std::uint64_t> r0, r1;
  EXPECT_FALSE(ModulusSwitchRecover(wrong, 4, 64, 8, 8, &r0, &r1));
}

TEST(YpirModulusSwitchTest, Recover_NullOut_ReturnsFalse) {
  std::vector<std::uint8_t> ct(3, 0);
  std::vector<std::uint64_t> r0;
  EXPECT_FALSE(ModulusSwitchRecover(ct, 4, 64, 8, 8, &r0, nullptr));
}

}  // namespace
}  // namespace primihub::pir::ypir
