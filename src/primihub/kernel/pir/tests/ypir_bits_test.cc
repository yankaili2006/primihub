/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_bits_test — verifies write_bits / read_bits / u64s↔bytes
 * port. The first three cases are direct ports of upstream's
 * `test_write_and_read_bits` (bits.rs line 92): they pin the exact
 * bit-level semantics at known offsets. The roundtrip case scales
 * to 42 14-bit values like upstream's test, but with a deterministic
 * seed so failures are reproducible.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_bits.h"

#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

// Upstream Rust test (bits.rs:101):
//   let mut buffer = [0u8; 4];
//   write_bits(&mut buffer, 0b11010101, 1, 6);
//   assert_eq!(buffer, [0b00101010, 0, 0, 0]);
//   let value = read_bits(&buffer, 1, 6);
//   assert_eq!(value, 0b010101);
TEST(YpirBitsTest, WriteBits_Offset1_6Bits_MatchesUpstream) {
  std::vector<std::uint8_t> buffer(4, 0u);
  WriteBits(buffer.data(), buffer.size(),
            /*val=*/0b11010101u, /*bit_offs=*/1, /*num_bits=*/6);
  EXPECT_EQ(buffer[0], 0b00101010u)
      << "Writing 0b11010101 with num_bits=6 takes low 6 bits "
      << "(0b010101) shifted left by 1 → 0b00101010.";
  EXPECT_EQ(buffer[1], 0u);
  EXPECT_EQ(buffer[2], 0u);
  EXPECT_EQ(buffer[3], 0u);

  std::uint64_t value = ReadBits(buffer.data(), buffer.size(),
                                  /*bit_offs=*/1, /*num_bits=*/6);
  EXPECT_EQ(value, 0b010101u);
}

// Upstream Rust test (bits.rs:108):
//   write_bits(&mut buffer2, 0b11111111, 0, 8);
//   assert_eq!(buffer2, [0b11111111, 0, 0, 0]);
//   let value2 = read_bits(&buffer2, 0, 8);
//   assert_eq!(value2, 0b11111111);
TEST(YpirBitsTest, WriteBits_WholeByteAlignedFromZero_MatchesUpstream) {
  std::vector<std::uint8_t> buffer(4, 0u);
  WriteBits(buffer.data(), buffer.size(), 0b11111111u, 0, 8);
  EXPECT_EQ(buffer[0], 0b11111111u);
  EXPECT_EQ(buffer[1], 0u);
  std::uint64_t value = ReadBits(buffer.data(), buffer.size(), 0, 8);
  EXPECT_EQ(value, 0b11111111u);
}

// Upstream Rust test (bits.rs:114):
//   write_bits(&mut buffer3, 0b10101010, 4, 4);
//   assert_eq!(buffer3, [0b10100000, 0, 0, 0]);
//   let value3 = read_bits(&buffer3, 4, 4);
//   assert_eq!(value3, 0b1010);
TEST(YpirBitsTest, WriteBits_HighNibble_MatchesUpstream) {
  std::vector<std::uint8_t> buffer(4, 0u);
  WriteBits(buffer.data(), buffer.size(), 0b10101010u, 4, 4);
  EXPECT_EQ(buffer[0], 0b10100000u);
  std::uint64_t value = ReadBits(buffer.data(), buffer.size(), 4, 4);
  EXPECT_EQ(value, 0b1010u);
}

// Upstream Rust test (bits.rs:122): pack 42 random 14-bit values and
// verify each can be read back. We use a deterministic mt19937_64 so
// failures are reproducible (upstream uses fastrand which is also
// deterministic per-thread but not across runs in CI).
TEST(YpirBitsTest, Roundtrip_42x14Bits_DeterministicSeed) {
  constexpr std::size_t kNum = 42;
  constexpr std::size_t kBitsPer = 14;
  std::mt19937_64 rng(/*seed=*/0xC0FFEEu);
  std::vector<std::uint64_t> vals(kNum);
  for (std::size_t i = 0; i < kNum; ++i) {
    vals[i] = rng() % (static_cast<std::uint64_t>(1) << kBitsPer);
  }
  const std::size_t total_sz_bytes = (kNum * kBitsPer + 7u) / 8u;
  std::vector<std::uint8_t> buffer(total_sz_bytes, 0u);
  std::size_t bit_offs = 0;
  for (std::size_t i = 0; i < kNum; ++i) {
    WriteBits(buffer.data(), buffer.size(), vals[i], bit_offs, kBitsPer);
    bit_offs += kBitsPer;
  }
  for (std::size_t i = 0; i < kNum; ++i) {
    std::uint64_t got = ReadBits(buffer.data(), buffer.size(),
                                  i * kBitsPer, kBitsPer);
    EXPECT_EQ(got, vals[i])
        << "Roundtrip failure at index " << i << " — expected "
        << vals[i] << " got " << got;
  }
}

// U64sToContiguousBytes / ContiguousBytesToU64s — full-stack
// roundtrip via the convenience wrappers, not WriteBits/ReadBits
// directly.
TEST(YpirBitsTest, U64sToBytes_Roundtrip_13BitsPer_DeterministicSeed) {
  constexpr std::size_t kNum = 17;
  constexpr std::size_t kBits = 13;
  std::mt19937_64 rng(/*seed=*/0xDECAFu);
  std::vector<std::uint64_t> vals(kNum);
  for (std::size_t i = 0; i < kNum; ++i) {
    vals[i] = rng() % (static_cast<std::uint64_t>(1) << kBits);
  }
  auto packed = U64sToContiguousBytes(vals, kBits);
  EXPECT_EQ(packed.size(), (kNum * kBits + 7u) / 8u);
  auto recovered = ContiguousBytesToU64s(packed, kBits);
  // ContiguousBytesToU64s recovers floor(packed.size() * 8 / kBits)
  // values, which may be > kNum (the packed bytes have some slack).
  // First kNum must match.
  ASSERT_GE(recovered.size(), kNum);
  for (std::size_t i = 0; i < kNum; ++i) {
    EXPECT_EQ(recovered[i], vals[i]) << "Mismatch at index " << i;
  }
}

TEST(YpirBitsTest, U64sToBytes_FullWidth64Bits) {
  // Edge: inp_mod_bits == 64 means each value packs into 8 bytes.
  // Exercises the 64-bit bitmask branch in WriteBits.
  const std::vector<std::uint64_t> vals = {
      0xFFFFFFFFFFFFFFFFULL,
      0xDEADBEEFCAFEBABEULL,
      0,
  };
  auto packed = U64sToContiguousBytes(vals, 64);
  EXPECT_EQ(packed.size(), vals.size() * 8u);
  auto recovered = ContiguousBytesToU64s(packed, 64);
  EXPECT_EQ(recovered.size(), vals.size());
  for (std::size_t i = 0; i < vals.size(); ++i) {
    EXPECT_EQ(recovered[i], vals[i]) << "64-bit roundtrip mismatch at " << i;
  }
}

TEST(YpirBitsTest, ReadBits_PreconditionViolations_ReturnZero) {
  // num_bits == 0 and num_bits > 64 must return 0 (upstream panics;
  // we keep a soft boundary). Verifies the wrapper contract is
  // exercised — if a future port replaces these with assertions
  // (e.g., to debug a caller bug), this test catches the change.
  std::vector<std::uint8_t> buf = {0xAB, 0xCD};
  EXPECT_EQ(ReadBits(buf.data(), buf.size(), 0, 0), 0u);
  EXPECT_EQ(ReadBits(buf.data(), buf.size(), 0, 65), 0u);
  EXPECT_EQ(ReadBits(nullptr, 0, 0, 8), 0u);
}

TEST(YpirBitsTest, WriteBits_StopsAtBufferEnd) {
  // Upstream's `while ... && byte_index < data.len()` silently stops
  // when the buffer is exhausted. Verifies our port does the same
  // (vs e.g. undefined-behavior writes past the end).
  std::vector<std::uint8_t> buf(1, 0u);
  // Write 16 bits but buffer is 1 byte — upper 8 bits should be
  // dropped silently.
  WriteBits(buf.data(), buf.size(), 0xFFFFu, 0, 16);
  EXPECT_EQ(buf[0], 0xFFu) << "Low 8 bits of 0xFFFF kept; upper 8 dropped";
}

TEST(YpirBitsTest, U64sToBytes_InvalidMod_ReturnsEmpty) {
  EXPECT_TRUE(U64sToContiguousBytes({1, 2, 3}, 0).empty());
  EXPECT_TRUE(U64sToContiguousBytes({1, 2, 3}, 65).empty());
  EXPECT_TRUE(ContiguousBytesToU64s({1, 2, 3}, 0).empty());
  EXPECT_TRUE(ContiguousBytesToU64s({1, 2, 3}, 65).empty());
}

}  // namespace
}  // namespace primihub::pir::ypir
