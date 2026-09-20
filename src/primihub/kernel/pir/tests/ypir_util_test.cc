/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_util_test — verifies the three Spiral-free util.rs helpers.
 * The cornerstone case is
 * NegacyclicMatrix_ColumnsEqualPermPerColumn — by definition the
 * i-th column of the matrix is NegacyclicPermU64Mod(a, i, modulus),
 * so the column-by-column equivalence is the strongest correctness
 * check we can do without a Spiral-bound reference.
 *
 * The U32-vs-U64 cross-consistency check leverages chunk 6a:
 * when modulus = 2^32, NegacyclicMatrixU64Mod must agree with
 * NegacyclicMatrixU32 from ypir_negacyclic.h after casting up.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

namespace primihub::pir::ypir {
namespace {

TEST(YpirUtilTest, NegacyclicPerm_Shift0_n4_KnownPattern) {
  // shift=0: first loop covers only i=0 -> out[0] = a[0].
  // Second loop i in 1..n: out[i] = modulus - (a[n - i] % modulus).
  // With a = [7, 11, 13, 17], modulus = 100:
  //   out[0] = 7
  //   out[1] = 100 - (a[3] % 100) = 100 - 17 = 83
  //   out[2] = 100 - (a[2] % 100) = 100 - 13 = 87
  //   out[3] = 100 - (a[1] % 100) = 100 - 11 = 89
  auto out = NegacyclicPermU64Mod({7, 11, 13, 17}, /*shift=*/0,
                                    /*modulus=*/100);
  const std::vector<std::uint64_t> expected = {7, 83, 87, 89};
  EXPECT_EQ(out, expected);
}

TEST(YpirUtilTest, NegacyclicPerm_Shift2_n4_KnownPattern) {
  // shift=2: first loop i in 0..3 -> out[0]=a[2], out[1]=a[1], out[2]=a[0].
  // Second loop i=3: out[3] = modulus - (a[n - (i - shift)] % modulus)
  //                       = modulus - a[4 - 1] = modulus - a[3]
  // With a = [7, 11, 13, 17], modulus = 100:
  //   out = [13, 11, 7, 100 - 17] = [13, 11, 7, 83]
  auto out = NegacyclicPermU64Mod({7, 11, 13, 17}, /*shift=*/2,
                                    /*modulus=*/100);
  const std::vector<std::uint64_t> expected = {13, 11, 7, 83};
  EXPECT_EQ(out, expected);
}

TEST(YpirUtilTest, NegacyclicPerm_ModulusClamp_ToZero) {
  // When a[*] % modulus == 0, modulus - 0 == modulus, which the
  // upstream clamps to 0. With a[i] = modulus (so a[i] % modulus = 0),
  // shift=0, the clamped slots must be 0.
  const std::uint64_t mod = 13;
  // a[0]=0 directly; a[1]=mod, a[2]=2*mod, a[3]=3*mod each %mod == 0.
  std::vector<std::uint64_t> a = {0, mod, 2 * mod, 3 * mod};
  auto out = NegacyclicPermU64Mod(a, /*shift=*/0, /*modulus=*/mod);
  // out[0] = a[0] = 0; out[1..3] use modulus clamp -> all 0.
  EXPECT_EQ(out, std::vector<std::uint64_t>({0, 0, 0, 0}));
}

TEST(YpirUtilTest, NegacyclicPerm_Empty_NoAllocation) {
  auto out = NegacyclicPermU64Mod({}, 0, 100);
  EXPECT_TRUE(out.empty());
}

TEST(YpirUtilTest, NegacyclicPerm_ShiftOutOfRange_ReturnsEmpty) {
  // Upstream Rust would panic on a[shift - i] when shift >= n;
  // our wrapper returns empty.
  auto out = NegacyclicPermU64Mod({1, 2, 3}, /*shift=*/3,
                                    /*modulus=*/100);
  EXPECT_TRUE(out.empty());
}

TEST(YpirUtilTest, NegacyclicMatrix_ColumnsEqualPerm_Cornerstone) {
  // The defining identity: column i of the matrix == perm(a, i).
  // This is the strongest invariant we can pin without a Spiral
  // reference; any port bug in either function fails this.
  const std::vector<std::uint64_t> a = {7, 11, 13, 17, 19};
  const std::uint64_t mod = 251;  // small prime keeps numbers readable
  const std::size_t n = a.size();
  auto m = NegacyclicMatrixU64Mod(a, mod);
  ASSERT_EQ(m.size(), n * n);
  for (std::size_t i = 0; i < n; ++i) {
    auto perm_i = NegacyclicPermU64Mod(a, i, mod);
    ASSERT_EQ(perm_i.size(), n);
    for (std::size_t j = 0; j < n; ++j) {
      // Column-major layout: cell (i, j) at index j*n + i.
      EXPECT_EQ(m[j * n + i], perm_i[j])
          << "Column " << i << " row " << j << " mismatch";
    }
  }
}

TEST(YpirUtilTest, NegacyclicMatrix_Empty_NoAllocation) {
  auto m = NegacyclicMatrixU64Mod({}, 100);
  EXPECT_TRUE(m.empty());
}

TEST(YpirUtilTest, NegacyclicMatrix_Mod2pow32_AgreesWithU32Version) {
  // Cross-consistency: modulus = 2^32 corresponds exactly to the
  // u32 wrapping-arithmetic version from chunk 6a. Build the same
  // matrix via both routes and verify byte-for-byte agreement
  // after u32→u64 cast.
  const std::uint64_t mod_2pow32 = static_cast<std::uint64_t>(1) << 32;
  const std::vector<std::uint32_t> a_u32 = {5, 6, 7, 8};
  std::vector<std::uint64_t> a_u64(a_u32.begin(), a_u32.end());

  auto m_u64 = NegacyclicMatrixU64Mod(a_u64, mod_2pow32);
  auto m_u32 = NegacyclicMatrixU32(a_u32);
  ASSERT_EQ(m_u64.size(), m_u32.size());
  for (std::size_t i = 0; i < m_u32.size(); ++i) {
    EXPECT_EQ(m_u64[i], static_cast<std::uint64_t>(m_u32[i]))
        << "u32 / u64 negacyclic matrix disagree at index " << i
        << ". The u32 version is the wrap-arithmetic specialisation "
        << "of the u64 version at modulus=2^32; any divergence here "
        << "means one of the ports drifted.";
  }
}

TEST(YpirUtilTest, ConcatHorizontal_3matrices_2x2_KnownLayout) {
  // Three 2×2 matrices A, B, C concatenated horizontally to a 2×6.
  // Upstream layout: out cell (i, k*a_cols + j) = v_a[k][i*a_cols + j].
  // For row 0 the values appear as [A00, A01, B00, B01, C00, C01].
  const std::vector<std::uint64_t> a_mat = {1, 2, 3, 4};       // [[1,2],[3,4]]
  const std::vector<std::uint64_t> b_mat = {10, 20, 30, 40};   // [[10,20],[30,40]]
  const std::vector<std::uint64_t> c_mat = {100, 200, 300, 400}; // [[100,200],[300,400]]
  auto out = ConcatHorizontalU64({a_mat, b_mat, c_mat}, 2, 2);
  // Expected row-major 2×6:
  //   row 0: 1 2 10 20 100 200
  //   row 1: 3 4 30 40 300 400
  const std::vector<std::uint64_t> expected = {
      1, 2, 10, 20, 100, 200,
      3, 4, 30, 40, 300, 400,
  };
  EXPECT_EQ(out, expected);
}

TEST(YpirUtilTest, ConcatHorizontal_WrongSize_ReturnsEmpty) {
  // 2x2 expected but second matrix has 3 entries -> empty.
  auto out = ConcatHorizontalU64({{1, 2, 3, 4}, {5, 6, 7}}, 2, 2);
  EXPECT_TRUE(out.empty());
}

TEST(YpirUtilTest, ConcatHorizontal_EmptyInput_ReturnsEmpty) {
  EXPECT_TRUE(ConcatHorizontalU64({}, 0, 0).empty());
  EXPECT_TRUE(ConcatHorizontalU64({{1, 2}}, 0, 2).empty());
}

}  // namespace
}  // namespace primihub::pir::ypir
