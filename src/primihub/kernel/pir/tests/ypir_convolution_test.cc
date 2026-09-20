/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_convolution_test — chunk 6 (naive reference helpers). These are
 * the oracles the optimised NTT convolution will be verified against.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirConvolutionTest, NaiveNegacyclic_HandComputed) {
  // (1 + 2X)(3 + 4X) mod (X^2 + 1) = 3 + 10X + 8X^2 = (3 - 8) + 10X
  // = -5 + 10X. -5 wraps to 2^32 - 5.
  const auto r = NaiveNegacyclicConvolve({1u, 2u}, {3u, 4u});
  ASSERT_EQ(r.size(), 2u);
  EXPECT_EQ(r[0], 4294967291u);  // -5 mod 2^32
  EXPECT_EQ(r[1], 10u);
}

TEST(YpirConvolutionTest, NaiveNegacyclic_IdentityIsNoOp) {
  // Convolving with [1, 0, ...] (the multiplicative identity) returns a.
  const std::vector<std::uint32_t> a = {5u, 7u, 11u, 13u};
  std::vector<std::uint32_t> id(a.size(), 0u);
  id[0] = 1u;
  EXPECT_EQ(NaiveNegacyclicConvolve(a, id), a);
}

TEST(YpirConvolutionTest, NaiveNegacyclic_WrapsModulo2Pow32) {
  // Large coefficients exercise u32 wraparound in the products/sums.
  const auto r =
      NaiveNegacyclicConvolve({0xFFFFFFFFu, 0x80000000u}, {2u, 3u});
  // res[0] = 0xFFFFFFFF*2 + 0x80000000*(-3)  (mod 2^32)
  // res[1] = 0xFFFFFFFF*3 + 0x80000000*2
  EXPECT_EQ(r[0], (0xFFFFFFFFu * 2u) + (0x80000000u * (0u - 3u)));
  EXPECT_EQ(r[1], (0xFFFFFFFFu * 3u) + (0x80000000u * 2u));
}

TEST(YpirConvolutionTest, NaiveMultiplyMatrices_NonTransposed) {
  // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
  const auto r = NaiveMultiplyMatrices({1u, 2u, 3u, 4u}, 2, 2,
                                       {5u, 6u, 7u, 8u}, 2, 2,
                                       /*is_b_transposed=*/false);
  EXPECT_EQ(r, (std::vector<std::uint32_t>{19u, 22u, 43u, 50u}));
}

TEST(YpirConvolutionTest, NaiveMultiplyMatrices_Transposed) {
  // Same product, but b stored transposed (columns contiguous):
  // b = [[5,6],[7,8]] -> b_t = {5,7, 6,8}.
  const auto r = NaiveMultiplyMatrices({1u, 2u, 3u, 4u}, 2, 2,
                                       {5u, 7u, 6u, 8u}, 2, 2,
                                       /*is_b_transposed=*/true);
  EXPECT_EQ(r, (std::vector<std::uint32_t>{19u, 22u, 43u, 50u}));
}

}  // namespace
}  // namespace primihub::pir::ypir
