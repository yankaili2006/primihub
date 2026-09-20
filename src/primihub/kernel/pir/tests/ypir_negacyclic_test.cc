/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_negacyclic_test — verifies the three Spiral-free negacyclic
 * helpers ported in chunk 6a. The cornerstone case is
 * MatrixVsConvolve_Equivalence: M = NegacyclicMatrixU32(b) is the
 * matrix form of negacyclic convolution by b, so M·a (column vector)
 * must equal NaiveNegacyclicConvolveU32(a, b). The two functions
 * are independent implementations of the same math, so the
 * equivalence test catches any port bug in either.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

// Helper: multiply column-major matrix M (stored as `M[j*n + i]`)
// by column vector `a`, all u32 wrapping arithmetic. Mirrors the
// way upstream consumers (LWE encrypt_many) use the matrix.
std::vector<std::uint32_t> MulMatVecU32(
    const std::vector<std::uint32_t>& m_col_major,
    const std::vector<std::uint32_t>& a, std::size_t n) {
  std::vector<std::uint32_t> out(n, 0u);
  for (std::size_t i = 0; i < n; ++i) {
    std::uint32_t sum = 0u;
    for (std::size_t j = 0; j < n; ++j) {
      // Column-major: cell (i, j) is at `j * n + i`.
      sum = sum + m_col_major[j * n + i] * a[j];
    }
    out[i] = sum;
  }
  return out;
}

TEST(YpirNegacyclicTest, NegacyclicPerm_n4_KnownPattern) {
  // a = [a0, a1, a2, a3]; res[0] = a0; res[i] = -a[(n-i) mod n].
  // i=1 -> -a[3]; i=2 -> -a[2]; i=3 -> -a[1].
  const std::vector<std::uint32_t> a = {7, 11, 13, 17};
  auto out = NegacyclicPermU32(a);
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0], 7u);
  EXPECT_EQ(out[1], static_cast<std::uint32_t>(0u - 17u));  // = 2^32 - 17
  EXPECT_EQ(out[2], static_cast<std::uint32_t>(0u - 13u));
  EXPECT_EQ(out[3], static_cast<std::uint32_t>(0u - 11u));
}

TEST(YpirNegacyclicTest, NegacyclicPerm_Empty_NoCrash) {
  auto out = NegacyclicPermU32({});
  EXPECT_TRUE(out.empty());
}

TEST(YpirNegacyclicTest, NegacyclicMatrix_n1_Identity) {
  // n=1: matrix is just [b[0]]. By the formula M[0]=b[0]
  // (i=0, j=0; (n+i-j)%n = 0; i<j false).
  const std::vector<std::uint32_t> b = {42};
  auto m = NegacyclicMatrixU32(b);
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0], 42u);
}

TEST(YpirNegacyclicTest, NegacyclicMatrix_n2_KnownPattern) {
  // n=2, b=[b0, b1]. Compute M[j*2 + i]:
  //   (0,0): b[(2+0-0)%2]=b[0]; i<j? no -> b0
  //   (0,1): b[(2+0-1)%2]=b[1]; i<j? yes -> -b1
  //   (1,0): b[(2+1-0)%2]=b[1]; i<j? no -> b1
  //   (1,1): b[(2+1-1)%2]=b[0]; i<j? no -> b0
  // Column-major layout: M = [(0,0), (1,0), (0,1), (1,1)] = [b0, b1, -b1, b0]
  const std::vector<std::uint32_t> b = {5, 7};
  auto m = NegacyclicMatrixU32(b);
  ASSERT_EQ(m.size(), 4u);
  EXPECT_EQ(m[0], 5u);                                       // (0,0)
  EXPECT_EQ(m[1], 7u);                                       // (1,0)
  EXPECT_EQ(m[2], static_cast<std::uint32_t>(0u - 7u));      // (0,1) = -b1
  EXPECT_EQ(m[3], 5u);                                       // (1,1)
}

TEST(YpirNegacyclicTest, NegacyclicMatrix_Empty_NoAllocation) {
  auto m = NegacyclicMatrixU32({});
  EXPECT_TRUE(m.empty());
}

TEST(YpirNegacyclicTest, NaiveConvolve_LengthMismatch_ReturnsEmpty) {
  auto out = NaiveNegacyclicConvolveU32({1, 2, 3}, {1, 2});
  EXPECT_TRUE(out.empty());
}

TEST(YpirNegacyclicTest, NaiveConvolve_DeltaB_ReturnsA) {
  // b = [1, 0, 0, 0] (delta at 0) — convolution with a delta returns a.
  // Walk through the formula at i=0:
  //   j=0: b[0]=1, i<j no, contributes a[0]*1 = a[0]
  //   j=1: b[(4+0-1)%4]=b[3]=0, contributes 0
  //   j=2: b[(4+0-2)%4]=b[2]=0
  //   j=3: b[(4+0-3)%4]=b[1]=0
  // So res[0] = a[0]. Similar logic gives res[i] = a[i].
  const std::vector<std::uint32_t> a = {10, 20, 30, 40};
  const std::vector<std::uint32_t> b = {1, 0, 0, 0};
  auto out = NaiveNegacyclicConvolveU32(a, b);
  EXPECT_EQ(out, a);
}

TEST(YpirNegacyclicTest, MatrixVsConvolve_Equivalence_n4) {
  // The canonical correctness check: build M = matrix(b), then
  // M·a column-mul must equal naive_convolve(a, b). This catches
  // any port bug in either function — they're independent
  // implementations of the same math.
  const std::vector<std::uint32_t> a = {1, 2, 3, 4};
  const std::vector<std::uint32_t> b = {5, 6, 7, 8};
  auto m = NegacyclicMatrixU32(b);
  ASSERT_EQ(m.size(), 16u);
  auto via_matrix = MulMatVecU32(m, a, 4);
  auto via_convolve = NaiveNegacyclicConvolveU32(a, b);
  EXPECT_EQ(via_matrix, via_convolve)
      << "NegacyclicMatrixU32 and NaiveNegacyclicConvolveU32 must "
      << "produce the same result when the matrix is multiplied by a "
      << "as a column. If this fails, one of the two ports drifted.";
}

TEST(YpirNegacyclicTest, MatrixVsConvolve_Equivalence_n16_LargeRange) {
  // Larger n with values approaching u32 wrap boundaries — exercises
  // wrap-around arithmetic that small cases don't reach.
  const std::size_t n = 16;
  std::vector<std::uint32_t> a(n), b(n);
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = static_cast<std::uint32_t>(0xFFFFFF00u + i);  // near 2^32
    b[i] = static_cast<std::uint32_t>(0xABCDEF00u + i);
  }
  auto m = NegacyclicMatrixU32(b);
  auto via_matrix = MulMatVecU32(m, a, n);
  auto via_convolve = NaiveNegacyclicConvolveU32(a, b);
  EXPECT_EQ(via_matrix, via_convolve);
}

}  // namespace
}  // namespace primihub::pir::ypir
