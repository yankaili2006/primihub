/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_matrices_test — verifies the deterministic, zero-PRNG
 * subset of utils::matrices. Three operations under test:
 *   * SwapMatrixFmt: row<->column transpose, hand-computed 2x3
 *     reference + 1x1 / wide row edge cases.
 *   * GetMatrixSecondAt: column extract equivalence with
 *     SwapMatrixFmt + out-of-range / empty soft boundaries.
 *   * VecMultU32U32: wrapping dot product, 4 cases including
 *     u32 overflow (forces wrap), size-mismatch error path with
 *     diagnostic referencing ErrorUnexpectedInputSize.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::frodo {
namespace {

TEST(FrodoMatricesTest, SwapMatrixFmt_2x3_HandComputed) {
  // [[1,2,3], [4,5,6]]  swap-> [[1,4], [2,5], [3,6]]
  const std::vector<std::vector<std::uint32_t>> in = {
      {1u, 2u, 3u},
      {4u, 5u, 6u},
  };
  const std::vector<std::vector<std::uint32_t>> expected = {
      {1u, 4u},
      {2u, 5u},
      {3u, 6u},
  };
  EXPECT_EQ(SwapMatrixFmt(in), expected);
}

TEST(FrodoMatricesTest, SwapMatrixFmt_DoubleSwapIsIdentity) {
  // Random 4x5 matrix, deterministic seed.
  std::mt19937_64 rng(/*seed=*/0xC0FFEE5EEDu);
  std::vector<std::vector<std::uint32_t>> m(4);
  for (auto& row : m) {
    row.resize(5);
    for (auto& v : row) v = static_cast<std::uint32_t>(rng() & 0xFFFFFFFFu);
  }
  const auto roundtrip = SwapMatrixFmt(SwapMatrixFmt(m));
  EXPECT_EQ(roundtrip, m);
}

TEST(FrodoMatricesTest, SwapMatrixFmt_EmptyInput_ReturnsEmpty) {
  // Soft boundary — upstream would panic on `matrix[0]`.
  const std::vector<std::vector<std::uint32_t>> in;
  EXPECT_TRUE(SwapMatrixFmt(in).empty());
}

TEST(FrodoMatricesTest, SwapMatrixFmt_1x4_BecomesColumn) {
  const std::vector<std::vector<std::uint32_t>> in = {{10u, 20u, 30u, 40u}};
  const std::vector<std::vector<std::uint32_t>> expected = {
      {10u}, {20u}, {30u}, {40u},
  };
  EXPECT_EQ(SwapMatrixFmt(in), expected);
}

TEST(FrodoMatricesTest, GetMatrixSecondAt_2x3_MatchesSwap) {
  const std::vector<std::vector<std::uint32_t>> in = {
      {1u, 2u, 3u},
      {4u, 5u, 6u},
  };
  // Column 0 = {1,4}; column 1 = {2,5}; column 2 = {3,6}.
  EXPECT_EQ(GetMatrixSecondAt(in, 0),
            std::vector<std::uint32_t>({1u, 4u}));
  EXPECT_EQ(GetMatrixSecondAt(in, 1),
            std::vector<std::uint32_t>({2u, 5u}));
  EXPECT_EQ(GetMatrixSecondAt(in, 2),
            std::vector<std::uint32_t>({3u, 6u}));
}

TEST(FrodoMatricesTest, GetMatrixSecondAt_AgreesWithSwap_RandomSeeded) {
  std::mt19937_64 rng(/*seed=*/0x5A11AD5u);
  std::vector<std::vector<std::uint32_t>> m(6);
  for (auto& row : m) {
    row.resize(7);
    for (auto& v : row) v = static_cast<std::uint32_t>(rng() & 0xFFFFFFFFu);
  }
  const auto swapped = SwapMatrixFmt(m);
  for (std::size_t i = 0; i < 7; ++i) {
    EXPECT_EQ(GetMatrixSecondAt(m, i), swapped[i])
        << "column " << i << " mismatch";
  }
}

TEST(FrodoMatricesTest, GetMatrixSecondAt_OutOfRange_ReturnsEmpty) {
  const std::vector<std::vector<std::uint32_t>> in = {{1u, 2u}};
  EXPECT_TRUE(GetMatrixSecondAt(in, 2).empty());
  EXPECT_TRUE(GetMatrixSecondAt(in, 100).empty());
}

TEST(FrodoMatricesTest, GetMatrixSecondAt_EmptyInput_ReturnsEmpty) {
  EXPECT_TRUE(GetMatrixSecondAt({}, 0).empty());
}

TEST(FrodoMatricesTest, VecMultU32U32_SmallHandComputed) {
  // {1,2,3} . {4,5,6} = 4 + 10 + 18 = 32.
  const std::vector<std::uint32_t> a = {1u, 2u, 3u};
  const std::vector<std::uint32_t> b = {4u, 5u, 6u};
  std::uint32_t out = 0;
  std::string err;
  ASSERT_EQ(VecMultU32U32(a, b, &out, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(out, 32u);
}

TEST(FrodoMatricesTest, VecMultU32U32_ZeroLengthIsZero) {
  std::uint32_t out = 12345u;
  std::string err;
  ASSERT_EQ(VecMultU32U32({}, {}, &out, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(out, 0u);
}

TEST(FrodoMatricesTest, VecMultU32U32_OverflowWraps) {
  // 0xFFFFFFFF * 1 + 1 * 0xFFFFFFFF = 0xFFFFFFFE (wraps).
  // Easier hand-check: {2^31, 2} . {2, 0} = 2^32 + 0 = 0 (wraps).
  const std::vector<std::uint32_t> a = {0x80000000u, 2u};
  const std::vector<std::uint32_t> b = {2u, 0u};
  std::uint32_t out = 12345u;
  std::string err;
  ASSERT_EQ(VecMultU32U32(a, b, &out, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(out, 0u) << "wrapping_mul(2^31, 2) = 0 (mod 2^32)";
}

TEST(FrodoMatricesTest, VecMultU32U32_AdditiveWrap) {
  // {0xFFFFFFFF, 1} . {1, 1} = 0xFFFFFFFF + 1 = 0 (wraps).
  const std::vector<std::uint32_t> a = {0xFFFFFFFFu, 1u};
  const std::vector<std::uint32_t> b = {1u, 1u};
  std::uint32_t out = 12345u;
  std::string err;
  ASSERT_EQ(VecMultU32U32(a, b, &out, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(out, 0u) << "wrapping_add(0xFFFFFFFF, 1) = 0 (mod 2^32)";
}

TEST(FrodoMatricesTest, VecMultU32U32_SizeMismatch_FailsWithMessage) {
  std::uint32_t out = 0;
  std::string err;
  EXPECT_EQ(VecMultU32U32({1u, 2u, 3u}, {1u, 2u}, &out, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("row_len: 3"), std::string::npos) << err;
  EXPECT_NE(err.find("col_len: 2"), std::string::npos) << err;
  EXPECT_NE(err.find("ErrorUnexpectedInputSize"), std::string::npos)
      << "diagnostic should reference upstream error type for "
      << "cross-doc traceability; got: " << err;
}

TEST(FrodoMatricesTest, VecMultU32U32_NullOut_FailsWithMessage) {
  std::string err;
  EXPECT_EQ(VecMultU32U32({1u}, {1u}, nullptr, &err), retcode::FAIL);
  EXPECT_NE(err.find("out must be non-null"), std::string::npos) << err;
}

}  // namespace
}  // namespace primihub::pir::frodo
