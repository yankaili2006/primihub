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



// ---- Chunk 2b-i tests (GenerateLweMatrixFromSeed) --------------

namespace {
SeedBytes IotaSeed(std::uint8_t start) {
  SeedBytes s;
  for (std::size_t i = 0; i < 32; ++i) {
    s[i] = static_cast<std::uint8_t>(start + i);
  }
  return s;
}
}  // namespace

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeed_Shape) {
  const auto seed = IotaSeed(0);
  const std::size_t lwe_dim = 13;
  const std::size_t width = 7;
  const auto a = GenerateLweMatrixFromSeed(seed, lwe_dim, width);
  ASSERT_EQ(a.size(), width);
  for (const auto& col : a) {
    EXPECT_EQ(col.size(), lwe_dim);
  }
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeed_Deterministic) {
  const auto seed = IotaSeed(42);
  const auto a = GenerateLweMatrixFromSeed(seed, 8, 5);
  const auto b = GenerateLweMatrixFromSeed(seed, 8, 5);
  EXPECT_EQ(a, b)
      << "same seed must produce identical matrices — required "
      << "for client/server matrix A agreement";
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeed_DifferentSeeds) {
  // Cross-seed independence — with 8*5=40 u32s drawn, the
  // probability that all match is ~2^-1280.
  const auto a = GenerateLweMatrixFromSeed(IotaSeed(0), 8, 5);
  const auto b = GenerateLweMatrixFromSeed(IotaSeed(1), 8, 5);
  EXPECT_NE(a, b)
      << "different seeds produced identical matrices — PRNG seed "
      << "derivation is broken";
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeed_ColumnOrderMatchesUpstream) {
  // Upstream fills column-by-column (outer loop is over `width`,
  // inner over `lwe_dim`). This test pins the iteration order:
  // generating a 1xN matrix must match the FIRST N u32s of the
  // SeededRng stream from the same seed.
  const auto seed = IotaSeed(7);
  SeededRng ref(seed);
  std::vector<std::uint32_t> expected_first_col;
  expected_first_col.reserve(6);
  for (int i = 0; i < 6; ++i) {
    expected_first_col.push_back(ref.NextU32());
  }
  const auto a = GenerateLweMatrixFromSeed(seed, /*lwe_dim=*/6,
                                            /*width=*/1);
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], expected_first_col)
      << "column-major iteration order does not match upstream";
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeedFlat_EmptyDims_ReturnsEmpty) {
  // chunk g-2 boundary: empty lwe_dim or width should return an
  // empty ColMajorMatrix, matching the per-column overload's
  // soft-boundary semantics. The downstream consumers (chunks
  // g-3..g-5) rely on this so the flat-vs-nested migration is
  // drop-in at any caller that handles empty inputs.
  const auto seed = IotaSeed(0);
  EXPECT_TRUE(GenerateLweMatrixFromSeedFlat(seed, 0, 5).empty());
  EXPECT_TRUE(GenerateLweMatrixFromSeedFlat(seed, 5, 0).empty());
  EXPECT_TRUE(GenerateLweMatrixFromSeedFlat(seed, 0, 0).empty());
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeedFlat_MatchesSeededRng_Small) {
  // chunk g-2 lower-level pin: the flat overload must produce
  // exactly the first `lwe_dim * width` u32s of the SeededRng
  // stream, in column-major order. Same shape of assertion as
  // chunk-2b-i ColumnOrderMatchesUpstream but against the new
  // overload.
  const auto seed = IotaSeed(11);
  const std::size_t lwe_dim = 6;
  const std::size_t width = 4;

  SeededRng ref(seed);
  std::vector<std::uint32_t> expected;
  expected.reserve(lwe_dim * width);
  for (std::size_t i = 0; i < lwe_dim * width; ++i) {
    expected.push_back(ref.NextU32());
  }

  const auto flat = GenerateLweMatrixFromSeedFlat(seed, lwe_dim, width);
  ASSERT_EQ(flat.width(), width);
  ASSERT_EQ(flat.height(), lwe_dim);
  for (std::size_t c = 0; c < width; ++c) {
    for (std::size_t r = 0; r < lwe_dim; ++r) {
      EXPECT_EQ(flat.at(c, r), expected[c * lwe_dim + r])
          << "flat overload byte stream mismatch at (col=" << c
          << ", row=" << r << ")";
    }
  }
}

TEST(FrodoMatricesTest, GenerateLweMatrixFromSeedFlat_MatchesPerColumn_Width2049) {
  // chunk g-2 BYTE-FOR-BYTE regression guard against the
  // per-column overload at a width that crosses the prior
  // chunk-d FillBytesBulk batching boundary (2049 > kColsPerBatch=1024
  // in the prior batched form). The two overloads must produce
  // the same column-major output -- this is the contract that
  // lets chunks g-3..g-5 migrate consumers from the nested form
  // to the flat form without changing observable behaviour at
  // any FrodoPIR shard or query.
  const auto seed = IotaSeed(5);
  const std::size_t lwe_dim = 16;
  const std::size_t width = 2049;

  const auto old_form = GenerateLweMatrixFromSeed(seed, lwe_dim, width);
  const auto flat = GenerateLweMatrixFromSeedFlat(seed, lwe_dim, width);

  ASSERT_EQ(old_form.size(), width);
  ASSERT_EQ(flat.width(), width);
  ASSERT_EQ(flat.height(), lwe_dim);
  for (std::size_t c = 0; c < width; ++c) {
    ASSERT_EQ(old_form[c].size(), lwe_dim)
        << "per-column overload produced wrong-size column at c=" << c;
    for (std::size_t r = 0; r < lwe_dim; ++r) {
      ASSERT_EQ(flat.at(c, r), old_form[c][r])
          << "flat vs per-column overload diverged at (col=" << c
          << ", row=" << r << ")";
    }
  }
}



// ---- Chunk 2c tests (RandomTernary + RandomTernaryVector) ------

TEST(FrodoMatricesTest, RandomTernary_OnlyValidValues) {
  // Every output must be in {0, 1, 0xFFFFFFFF}. Over 1024 trials
  // the probability of accidentally missing an invalid output is
  // dominated by an OsRng failure, not by the test design.
  for (int i = 0; i < 1024; ++i) {
    const std::uint32_t v = RandomTernary();
    const bool ok = (v == 0u) || (v == 1u) || (v == 0xFFFFFFFFu);
    EXPECT_TRUE(ok) << "RandomTernary returned " << v
                    << " — not in {0, 1, 0xFFFFFFFF}";
  }
}

TEST(FrodoMatricesTest, RandomTernary_AllThreeValuesAppear) {
  // Each value has roughly 1/3 probability; over 1024 trials the
  // probability of missing any one value is (2/3)^1024 ≈ 0.
  // Failing this test indicates a broken distribution.
  std::array<bool, 3> seen = {false, false, false};
  for (int i = 0; i < 1024; ++i) {
    const std::uint32_t v = RandomTernary();
    if (v == 0u) seen[0] = true;
    else if (v == 1u) seen[1] = true;
    else if (v == 0xFFFFFFFFu) seen[2] = true;
  }
  EXPECT_TRUE(seen[0]) << "value 0 never produced in 1024 trials";
  EXPECT_TRUE(seen[1]) << "value 1 never produced in 1024 trials";
  EXPECT_TRUE(seen[2])
      << "value 0xFFFFFFFF never produced in 1024 trials";
}

TEST(FrodoMatricesTest, RandomTernaryVector_SizeAndValues) {
  const std::size_t n = 128;
  const auto v = RandomTernaryVector(n);
  ASSERT_EQ(v.size(), n);
  for (std::size_t i = 0; i < n; ++i) {
    const bool ok = (v[i] == 0u) || (v[i] == 1u) ||
                    (v[i] == 0xFFFFFFFFu);
    EXPECT_TRUE(ok) << "elem " << i << " = " << v[i];
  }
}

TEST(FrodoMatricesTest, RandomTernaryVector_EmptyOk) {
  EXPECT_TRUE(RandomTernaryVector(0).empty());
}

// ---- AVX2 inner-kernel equivalence -------------------------------
//
// VecMultU32U32 dispatches to an AVX2 helper for length >= 16 on
// AVX2-capable hardware. These tests pin the SIMD path against
// independent scalar computation for sizes that straddle the
// 8-lane boundary, the dispatch threshold, and a typical
// FrodoPIR LWE-dim length (512).

namespace {

std::uint32_t ScalarVecMult(const std::vector<std::uint32_t>& a,
                            const std::vector<std::uint32_t>& b) {
  std::uint32_t acc = 0u;
  for (std::size_t i = 0; i < a.size(); ++i) acc += a[i] * b[i];
  return acc;
}

std::vector<std::uint32_t> Mt19937Vec(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint32_t> v(n);
  for (auto& x : v) x = static_cast<std::uint32_t>(rng());
  return v;
}

}  // namespace

TEST(FrodoMatricesTest, VecMultU32U32_SimdMatchesScalar_Below16) {
  // Below the n >= 16 dispatch threshold — exercises the scalar path
  // even on AVX2 hardware, ensuring the dispatch decision is right.
  for (std::size_t n : {1u, 7u, 8u, 9u, 15u}) {
    auto a = Mt19937Vec(n, 0xCAFE0001 + n);
    auto b = Mt19937Vec(n, 0xCAFE0002 + n);
    std::uint32_t got = 0u;
    std::string err;
    ASSERT_EQ(VecMultU32U32(a, b, &got, &err), retcode::SUCCESS) << err;
    EXPECT_EQ(got, ScalarVecMult(a, b)) << "mismatch at n=" << n;
  }
}

TEST(FrodoMatricesTest, VecMultU32U32_SimdMatchesScalar_BoundaryAndTail) {
  // Sizes that straddle the 8-lane boundary in the AVX2 main loop +
  // exercise the scalar tail of various widths.
  for (std::size_t n : {16u, 17u, 23u, 24u, 31u, 32u, 33u, 64u, 100u}) {
    auto a = Mt19937Vec(n, 0xDEAD0001 + n);
    auto b = Mt19937Vec(n, 0xDEAD0002 + n);
    std::uint32_t got = 0u;
    std::string err;
    ASSERT_EQ(VecMultU32U32(a, b, &got, &err), retcode::SUCCESS) << err;
    EXPECT_EQ(got, ScalarVecMult(a, b))
        << "SIMD diverges from scalar at n=" << n;
  }
}

TEST(FrodoMatricesTest, VecMultU32U32_SimdMatchesScalar_LweDim512) {
  // Typical FrodoPIR LWE-dim VecMult length.
  const std::size_t n = 512;
  auto a = Mt19937Vec(n, 0xBEEF0001);
  auto b = Mt19937Vec(n, 0xBEEF0002);
  std::uint32_t got = 0u;
  std::string err;
  ASSERT_EQ(VecMultU32U32(a, b, &got, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(got, ScalarVecMult(a, b));
}

TEST(FrodoMatricesTest, VecMultU32U32_SimdMatchesScalar_LargeWrappingValues) {
  // Pre-load values near 2^32 to exercise wrapping mul overflow.
  const std::size_t n = 64;
  std::vector<std::uint32_t> a(n), b(n);
  std::mt19937_64 rng(0xF00DBABE);
  for (std::size_t i = 0; i < n; ++i) {
    // 75% chance of upper-half values to force overflow paths.
    a[i] = (rng() % 4u == 0u) ? static_cast<std::uint32_t>(rng())
                              : static_cast<std::uint32_t>(0x80000000ULL
                                                            | rng());
    b[i] = (rng() % 4u == 0u) ? static_cast<std::uint32_t>(rng())
                              : static_cast<std::uint32_t>(0x80000000ULL
                                                            | rng());
  }
  std::uint32_t got = 0u;
  std::string err;
  ASSERT_EQ(VecMultU32U32(a, b, &got, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(got, ScalarVecMult(a, b));
}

}  // namespace
}  // namespace primihub::pir::frodo
