/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_convolution_ntt_test — chunk 6 pt2. Verifies the HEXL-backed
 * 2-modulus CRT-NTT Convolution against the naive oracle. The exact
 * (signed) convolution coefficient must stay within (-Q/2, Q/2] (Q ~
 * 2^56) for the u32 result to match the naive u32-wrapping reference,
 * so the convolution tests use a small second operand (as in YPIR's
 * use, where it is the gaussian secret key).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution_ntt.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution.h"

namespace primihub::pir::ypir {
namespace {

TEST(YpirConvolutionNttTest, RawOfNtt_RoundTripsIdentity) {
  // Raw(Ntt(a)) == a for arbitrary u32 a (a < 2^32 << Q).
  Convolution conv(8);
  const std::vector<std::uint32_t> a = {0xDEADBEEFu, 0x12345678u, 1000000u,
                                        0xFFFFFFFFu, 42u,        0x80000000u,
                                        7u,          999999u};
  EXPECT_EQ(conv.Raw(conv.Ntt(a)), a);
}

TEST(YpirConvolutionNttTest, Convolve_MatchesNaive_SmallSecondOperand) {
  Convolution conv(8);
  const std::vector<std::uint32_t> a = {0xDEADBEEFu, 0x12345678u, 1000000u,
                                        0xFFFFFFFFu, 42u,        0x80000000u,
                                        7u,          999999u};
  const std::vector<std::uint32_t> b = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
  EXPECT_EQ(conv.Convolve(a, b), NaiveNegacyclicConvolve(a, b));
}

TEST(YpirConvolutionNttTest, Convolve_BothSmallOperands_MatchesNaive) {
  // Both operands small -> the exact integer convolution is tiny, well
  // within (-Q/2, Q/2], so it equals the naive reference exactly.
  Convolution conv(8);
  const std::vector<std::uint32_t> a = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
  const std::vector<std::uint32_t> b = {2u, 0u, 1u, 0u, 3u, 0u, 0u, 1u};
  EXPECT_EQ(conv.Convolve(a, b), NaiveNegacyclicConvolve(a, b));
}

TEST(YpirConvolutionNttTest, Convolve_LargerN_MatchesNaive) {
  // n=16, full-u32 first operand x small second operand stays in domain.
  Convolution conv(16);
  std::vector<std::uint32_t> a(16), b(16);
  for (std::uint32_t i = 0; i < 16; ++i) {
    a[i] = 0x9E3779B9u * (i + 1u);  // spread across u32
    b[i] = (i * 37u) % 200u;        // small
  }
  EXPECT_EQ(conv.Convolve(a, b), NaiveNegacyclicConvolve(a, b));
}

TEST(YpirConvolutionNttTest, Convolve_IdentityIsNoOp) {
  Convolution conv(8);
  const std::vector<std::uint32_t> a = {5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u};
  std::vector<std::uint32_t> id(8, 0u);
  id[0] = 1u;
  EXPECT_EQ(conv.Convolve(a, id), a);
}

TEST(YpirConvolutionNttTest, NttShape_Is2N) {
  Convolution conv(16);
  std::vector<std::uint32_t> a(16, 3u);
  EXPECT_EQ(conv.Ntt(a).size(), 32u);
}

}  // namespace
}  // namespace primihub::pir::ypir
