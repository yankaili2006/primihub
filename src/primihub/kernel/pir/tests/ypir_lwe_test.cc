/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe_test — chunk 4c verification.
 *   * Hand-computed single encrypt/decrypt phase.
 *   * decrypt(encrypt(pt, e)) == pt + e roundtrip (incl. u32 wrapping).
 *   * encrypt_many: each column decrypts to v_pt[col] + e[col].
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirLweTest, Encrypt_HandComputed) {
  const std::vector<std::uint32_t> sk = {3, 5};
  const std::vector<std::uint32_t> a = {7, 11};
  // sum = 7*3 + 11*5 = 76; b = -76 + 100 + 2 = 26.
  const auto ct = LweEncrypt(sk, a, /*pt=*/100, /*e=*/2);
  ASSERT_EQ(ct.size(), 3u);
  EXPECT_EQ(ct[0], 7u);
  EXPECT_EQ(ct[1], 11u);
  EXPECT_EQ(ct[2], 26u);
}

TEST(YpirLweTest, Decrypt_RecoversPhase_PtPlusNoise) {
  const std::vector<std::uint32_t> sk = {3, 5, 9, 2};
  const std::vector<std::uint32_t> a = {7, 11, 4, 6};
  const std::uint32_t pt = 1000, e = 3;
  const auto ct = LweEncrypt(sk, a, pt, e);
  EXPECT_EQ(LweDecrypt(sk, ct), pt + e);
}

TEST(YpirLweTest, Roundtrip_U32Wrapping) {
  // Large sk / a / pt exercise u32 wraparound in the dot product and b.
  const std::vector<std::uint32_t> sk = {0xDEADBEEFu, 0x12345678u, 0xFFFFFFFFu};
  const std::vector<std::uint32_t> a = {0x80000001u, 0x7FFFFFFFu, 0xABCDEF01u};
  const std::uint32_t pt = 0xCAFEBABEu, e = 5;
  const auto ct = LweEncrypt(sk, a, pt, e);
  EXPECT_EQ(LweDecrypt(sk, ct), static_cast<std::uint32_t>(pt + e));
}

TEST(YpirLweTest, Decrypt_NoiselessRecoversPlaintext) {
  const std::vector<std::uint32_t> sk = {4, 8, 15, 16};
  const std::vector<std::uint32_t> a = {23, 42, 1, 99};
  const std::uint32_t pt = 12345;
  const auto ct = LweEncrypt(sk, a, pt, /*e=*/0);
  EXPECT_EQ(LweDecrypt(sk, ct), pt);
}

TEST(YpirLweTest, EncryptMany_EachColumnDecryptsToMessagePlusNoise) {
  const std::size_t n = 4;
  const std::vector<std::uint32_t> sk = {3, 5, 9, 2};
  const std::vector<std::uint32_t> a = {7, 11, 4, 6};
  const std::vector<std::uint32_t> v_pt = {100, 200, 300, 400};
  const std::vector<std::uint32_t> e = {1, 2, 3, 4};

  const auto ct = LweEncryptMany(sk, a, v_pt, e);
  ASSERT_EQ(ct.size(), n * n + n);

  for (std::size_t col = 0; col < n; ++col) {
    // Extract the col-th LWE sample: [nega_a[row*n+col] for row], last_row[col].
    std::vector<std::uint32_t> col_ct(n + 1);
    for (std::size_t row = 0; row < n; ++row) {
      col_ct[row] = ct[row * n + col];
    }
    col_ct[n] = ct[n * n + col];
    EXPECT_EQ(LweDecrypt(sk, col_ct), v_pt[col] + e[col]) << "col " << col;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
