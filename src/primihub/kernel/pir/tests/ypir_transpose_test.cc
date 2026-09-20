/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_transpose_test — unit tests for the YPIR transpose port.
 * The first case (Bytes_2x8) is a direct port of upstream
 * transpose.rs::test_transpose, validating that the C++ matches the
 * Rust reference output byte-for-byte. The remaining cases exercise
 * the generic / f64 paths that upstream covers only by call sites.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_transpose.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

// Upstream Rust:
//   let mut buf = vec![0u8; 16]; for i in 0..16 { buf[i] = i as u8; }
//   let out = transpose(&buf, 2, 8, 1);
//   assert_eq!(out, vec![0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]);
TEST(YpirTransposeTest, Bytes_2x8_MatchesUpstreamRust) {
  std::vector<uint8_t> buf(16);
  for (std::size_t i = 0; i < 16; ++i) {
    buf[i] = static_cast<uint8_t>(i);
  }
  auto out = Transpose(buf.data(), buf.size(), /*rows=*/2, /*cols=*/8,
                       /*bytes_per_pt_el=*/1);
  const std::vector<uint8_t> expected = {0, 8, 1, 9, 2, 10, 3,  11,
                                         4, 12, 5, 13, 6, 14, 7, 15};
  EXPECT_EQ(out, expected);
}

// Multi-byte elements: each element is 2 bytes wide, transpose should
// keep both bytes of an element together. Pattern: row 0 elems are
// {0xAA01, 0xAA02}; row 1 elems are {0xBB01, 0xBB02}. After transpose
// (2x2 → 2x2) we expect column 0 = {0xAA01, 0xBB01}, column 1 =
// {0xAA02, 0xBB02}. Little-endian byte layout.
TEST(YpirTransposeTest, Bytes_2x2_TwoBytePerElement) {
  std::vector<uint8_t> buf = {0x01, 0xAA, 0x02, 0xAA, 0x01, 0xBB, 0x02, 0xBB};
  auto out = Transpose(buf.data(), buf.size(), /*rows=*/2, /*cols=*/2,
                       /*bytes_per_pt_el=*/2);
  const std::vector<uint8_t> expected = {
      0x01, 0xAA, 0x01, 0xBB,  // column 0
      0x02, 0xAA, 0x02, 0xBB,  // column 1
  };
  EXPECT_EQ(out, expected);
}

// TransposeGeneric tiled path: 64 rows × 64 cols of uint32_t,
// element i*cols + j = i * 1000 + j. After transpose, position
// j * rows + i must equal i * 1000 + j.
TEST(YpirTransposeTest, Generic_64x64_TilePathCorrect) {
  constexpr std::size_t kRows = 64;
  constexpr std::size_t kCols = 64;
  std::vector<uint32_t> a(kRows * kCols);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      a[i * kCols + j] = static_cast<uint32_t>(i * 1000 + j);
    }
  }
  auto out = TransposeGeneric<uint32_t>(a, kRows, kCols);
  ASSERT_EQ(out.size(), kRows * kCols);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      EXPECT_EQ(out[j * kRows + i], static_cast<uint32_t>(i * 1000 + j))
          << "mismatch at (i=" << i << ", j=" << j << ")";
    }
  }
}

// TransposeGeneric fallback path: 7 × 5 doesn't divide 32; must
// fall back to tile=1 and still be correct.
TEST(YpirTransposeTest, Generic_7x5_FallbackPathCorrect) {
  constexpr std::size_t kRows = 7;
  constexpr std::size_t kCols = 5;
  std::vector<int> a(kRows * kCols);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      a[i * kCols + j] = static_cast<int>(i * 10 + j);
    }
  }
  auto out = TransposeGeneric<int>(a, kRows, kCols);
  ASSERT_EQ(out.size(), kRows * kCols);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      EXPECT_EQ(out[j * kRows + i], static_cast<int>(i * 10 + j));
    }
  }
}

// TransposeElems vector-of-vector flavour: 3 × 2 of std::string,
// non-trivially-copyable T.
TEST(YpirTransposeTest, Elems_3x2_StringMatrix) {
  const std::vector<std::vector<std::string>> in = {
      {"a", "b"},
      {"c", "d"},
      {"e", "f"},
  };
  auto out = TransposeElems<std::string>(in, /*rows=*/3, /*cols=*/2);
  ASSERT_EQ(out.size(), 2u);
  ASSERT_EQ(out[0].size(), 3u);
  EXPECT_EQ(out[0][0], "a");
  EXPECT_EQ(out[0][1], "c");
  EXPECT_EQ(out[0][2], "e");
  EXPECT_EQ(out[1][0], "b");
  EXPECT_EQ(out[1][1], "d");
  EXPECT_EQ(out[1][2], "f");
}

// TransposeF64 preconditions: must reject non-multiple-of-32 dims
// and report failure via false return (not abort).
TEST(YpirTransposeTest, F64_PreconditionFailure_ReturnsFalse) {
  std::vector<double> a(7 * 5, 1.0);
  std::vector<double> out(7 * 5, 99.0);
  EXPECT_FALSE(TransposeF64(out.data(), a.data(), 7, 5));
  // out must be untouched on precondition failure.
  for (double v : out) EXPECT_EQ(v, 99.0);
}

// TransposeF64 happy path: 32 × 64.
TEST(YpirTransposeTest, F64_32x64_TilePathCorrect) {
  constexpr std::size_t kRows = 32;
  constexpr std::size_t kCols = 64;
  std::vector<double> a(kRows * kCols);
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      a[i * kCols + j] = static_cast<double>(i) + 0.5 * static_cast<double>(j);
    }
  }
  std::vector<double> out(kRows * kCols, 0.0);
  ASSERT_TRUE(TransposeF64(out.data(), a.data(), kRows, kCols));
  for (std::size_t i = 0; i < kRows; ++i) {
    for (std::size_t j = 0; j < kCols; ++j) {
      EXPECT_DOUBLE_EQ(out[j * kRows + i],
                       static_cast<double>(i) + 0.5 * static_cast<double>(j));
    }
  }
}

// Empty / degenerate input must be safe.
TEST(YpirTransposeTest, Bytes_Empty_NoCrash) {
  auto out = Transpose(nullptr, 0, 0, 0, 0);
  EXPECT_TRUE(out.empty());
}

}  // namespace
}  // namespace primihub::pir::ypir
