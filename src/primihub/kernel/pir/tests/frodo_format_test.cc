/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_format_test — verifies the FrodoPIR bit-level format helper
 * port. Cornerstone tests:
 *   * U8 bit decomposition matches hand-computed LSB-first values.
 *   * U32 ↔ bits roundtrip via U32ToBitsLe + BitsToU32Le.
 *   * bytes ↔ bits roundtrip via BytesToBitsLe + BitsToBytesLe.
 *   * BitsToU32Le rejects oversized inputs with retcode::FAIL +
 *     diagnostic mentioning the upstream error type.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_format.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::frodo {
namespace {

TEST(FrodoFormatTest, U8ToBitsLe_Zero_AllZeros) {
  auto bits = U8ToBitsLe(0u);
  ASSERT_EQ(bits.size(), 8u);
  for (auto b : bits) EXPECT_EQ(b, 0u);
}

TEST(FrodoFormatTest, U8ToBitsLe_0xAB_LsbFirst) {
  // 0xAB = 1010_1011 (MSB-first); LSB-first the bits are
  // 1,1,0,1,0,1,0,1.
  auto bits = U8ToBitsLe(0xABu);
  const std::vector<std::uint8_t> expected = {1, 1, 0, 1, 0, 1, 0, 1};
  EXPECT_EQ(bits, expected);
}

TEST(FrodoFormatTest, U8ToBitsLe_0xFF_AllOnes) {
  auto bits = U8ToBitsLe(0xFFu);
  ASSERT_EQ(bits.size(), 8u);
  for (auto b : bits) EXPECT_EQ(b, 1u);
}

TEST(FrodoFormatTest, U32ToBitsLe_8Bits_LowByteOnly) {
  // x = 0xDEADBEEF; 8 LSB bits = 0xEF = 1110_1111 -> LSB-first
  // 1,1,1,1,0,1,1,1.
  auto bits = U32ToBitsLe(0xDEADBEEFu, 8);
  const std::vector<std::uint8_t> expected = {1, 1, 1, 1, 0, 1, 1, 1};
  EXPECT_EQ(bits, expected);
}

TEST(FrodoFormatTest, U32ToBitsLe_32Bits_FullExpansion) {
  // 0x00000001 LSB-first 32-bit -> first bit is 1, rest are 0.
  auto bits = U32ToBitsLe(1u, 32);
  ASSERT_EQ(bits.size(), 32u);
  EXPECT_EQ(bits[0], 1u);
  for (std::size_t i = 1; i < 32; ++i) EXPECT_EQ(bits[i], 0u);
}

TEST(FrodoFormatTest, U32ToBitsLe_OversizeBitLen_ClampsTo32) {
  // bit_len > 32 must clamp (upstream Rust panics; we keep the
  // soft boundary).
  auto bits = U32ToBitsLe(0xFFFFFFFFu, 100);
  EXPECT_EQ(bits.size(), 32u);
  for (auto b : bits) EXPECT_EQ(b, 1u);
}

TEST(FrodoFormatTest, BitsToBytesLe_HandComputed_3Bytes) {
  // 17 bits — should pack into 3 bytes (ceil(17/8)).
  // Pattern: bit 0 = 1, bit 8 = 1 (rest = 0)
  std::vector<std::uint8_t> bits(17, 0);
  bits[0] = 1;
  bits[8] = 1;
  auto bytes = BitsToBytesLe(bits);
  ASSERT_EQ(bytes.size(), 3u);
  EXPECT_EQ(bytes[0], 0b00000001u);  // bit 0 set
  EXPECT_EQ(bytes[1], 0b00000001u);  // bit 8 -> byte 1 bit 0
  EXPECT_EQ(bytes[2], 0u);
}

TEST(FrodoFormatTest, BytesToBitsLe_3Bytes_Pattern) {
  const std::vector<std::uint8_t> bytes = {0x01, 0xFF, 0x00};
  auto bits = BytesToBitsLe(bytes);
  ASSERT_EQ(bits.size(), 24u);
  EXPECT_EQ(bits[0], 1u);
  for (std::size_t i = 1; i < 8; ++i) EXPECT_EQ(bits[i], 0u);
  for (std::size_t i = 8; i < 16; ++i) EXPECT_EQ(bits[i], 1u);
  for (std::size_t i = 16; i < 24; ++i) EXPECT_EQ(bits[i], 0u);
}

TEST(FrodoFormatTest, BytesBitsRoundtrip_DeterministicSeed) {
  // Bytes -> bits -> bytes must be the identity (modulo the
  // ceil() padding when starting from bits of non-multiple-of-8
  // length, but starting from bytes there's no padding).
  std::mt19937_64 rng(/*seed=*/0xFEEDFACEu);
  std::vector<std::uint8_t> bytes(50);
  for (auto& b : bytes) b = static_cast<std::uint8_t>(rng() & 0xFFu);
  auto bits = BytesToBitsLe(bytes);
  auto recovered = BitsToBytesLe(bits);
  EXPECT_EQ(recovered, bytes);
}

TEST(FrodoFormatTest, BitsToU32Le_Roundtrip_FullWidth) {
  // U32ToBitsLe + BitsToU32Le must round-trip every u32 value
  // exactly. Spot-check a few key values.
  const std::vector<std::uint32_t> samples = {0u, 1u, 0xCAFEBABEu,
                                               0xFFFFFFFFu, 42u, 0x80000000u};
  for (std::uint32_t v : samples) {
    auto bits = U32ToBitsLe(v, 32);
    std::uint32_t out = 0;
    std::string err;
    ASSERT_EQ(BitsToU32Le(bits, &out, &err), retcode::SUCCESS)
        << "Failed roundtrip on " << v << ": " << err;
    EXPECT_EQ(out, v);
  }
}

TEST(FrodoFormatTest, BitsToU32Le_TooManyBits_FailsWithMessage) {
  // 33 bits packs into 5 bytes -> exceeds u32 size.
  std::vector<std::uint8_t> bits(33, 0);
  std::uint32_t out = 0;
  std::string err;
  EXPECT_EQ(BitsToU32Le(bits, &out, &err), retcode::FAIL);
  EXPECT_NE(err.find("too long to parse as u32"), std::string::npos)
      << err;
  EXPECT_NE(err.find("ErrorUnexpectedInputSize"), std::string::npos)
      << "Diagnostic should reference the upstream error type for "
      << "cross-doc traceability; got: " << err;
}

TEST(FrodoFormatTest, BitsToU32Le_NullOut_FailsWithMessage) {
  std::vector<std::uint8_t> bits = {1, 0, 1};
  std::string err;
  EXPECT_EQ(BitsToU32Le(bits, nullptr, &err), retcode::FAIL);
  EXPECT_NE(err.find("out must be non-null"), std::string::npos);
}

TEST(FrodoFormatTest, BytesFromU32Slice_KnownPattern) {
  // Two u32s with entry_bit_len = 4, total_bit_len = 6.
  // remainder = 6 % 4 = 2.
  // v = [0xF, 0x3]
  // first entry: U32ToBitsLe(0xF, 4) -> [1,1,1,1]
  // last entry:  U32ToBitsLe(0x3, 2) -> [1,1]
  // bits = [1,1,1,1, 1,1]; BitsToBytesLe -> [0b00111111] = 0x3F
  auto out = BytesFromU32Slice({0xFu, 0x3u}, /*entry_bit_len=*/4,
                                /*total_bit_len=*/6);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0], 0x3Fu);
}

TEST(FrodoFormatTest, BytesFromU32Slice_EmptyInput_ReturnsEmpty) {
  auto out = BytesFromU32Slice({}, 8, 8);
  EXPECT_TRUE(out.empty());
}

}  // namespace
}  // namespace primihub::pir::frodo
