/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_spiral_client chunk 12b-1: the arith/helper gaps
 * (InvertUintMod / MultiplyUintMod / SingleValue / MatrixWithIdentity /
 * GenTernaryMat). Pure scalar logic, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params P8() {
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, 1, 1, 1, 0,
                      0);
}

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirSpiralClientTest, InvertUintMod) {
  EXPECT_EQ(InvertUintMod(3, 7), 5u);   // hand-computed: 3*5=15==1 (mod 7)
  EXPECT_EQ(InvertUintMod(1, 7), 1u);
  EXPECT_EQ(InvertUintMod(0, 7), 0u);   // None sentinel
  EXPECT_EQ(InvertUintMod(7, 7), 0u);   // gcd != 1 -> None
  EXPECT_EQ(InvertUintMod(2, 8), 0u);   // gcd(2,8)=2 -> None

  const Params p = P8();
  for (std::uint64_t v : {2ull, 3ull, 2048ull, 65537ull, 1234567ull}) {
    const std::uint64_t inv = InvertUintMod(v, p.modulus);
    ASSERT_NE(inv, 0u) << "v=" << v;
    const std::uint64_t prod = static_cast<std::uint64_t>(
        (static_cast<__uint128_t>(v) * inv) % p.modulus);
    EXPECT_EQ(prod, 1u) << "v=" << v;
  }
}

TEST(YpirSpiralClientTest, MultiplyUintMod) {
  EXPECT_EQ(MultiplyUintMod(6, 7, 10), 2u);  // 42 % 10
  const std::uint64_t m = P8().modulus;
  const std::uint64_t a = m - 3, b = m - 5;
  const std::uint64_t expect = static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(a) * b) % m);
  EXPECT_EQ(MultiplyUintMod(a, b, m), expect);
}

TEST(YpirSpiralClientTest, SingleValue) {
  const Params p = P8();
  const PolyMatrixRaw r = SingleValue(p, 12345);
  EXPECT_EQ(r.rows, 1u);
  EXPECT_EQ(r.cols, 1u);
  ASSERT_EQ(r.data.size(), p.poly_len);
  EXPECT_EQ(r.data[0], 12345u);
  for (std::size_t z = 1; z < p.poly_len; ++z) EXPECT_EQ(r.data[z], 0u);
}

TEST(YpirSpiralClientTest, MatrixWithIdentity) {
  const Params p = P8();
  PolyMatrixRaw in;
  in.rows = 2;
  in.cols = 1;
  in.data.assign(2 * p.poly_len, 0);
  in.Poly(0, 0, p.poly_len)[0] = 11;
  in.Poly(1, 0, p.poly_len)[0] = 22;

  const PolyMatrixRaw r = MatrixWithIdentity(p, in);
  EXPECT_EQ(r.rows, 2u);
  EXPECT_EQ(r.cols, 3u);  // [in | I_2]
  // column 0 = in
  EXPECT_EQ(r.Poly(0, 0, p.poly_len)[0], 11u);
  EXPECT_EQ(r.Poly(1, 0, p.poly_len)[0], 22u);
  // columns 1..2 = identity (constant 1 on diagonal)
  EXPECT_EQ(r.Poly(0, 1, p.poly_len)[0], 1u);
  EXPECT_EQ(r.Poly(1, 1, p.poly_len)[0], 0u);
  EXPECT_EQ(r.Poly(0, 2, p.poly_len)[0], 0u);
  EXPECT_EQ(r.Poly(1, 2, p.poly_len)[0], 1u);
}

TEST(YpirSpiralClientTest, GenTernaryMatCountsAndDeterminism) {
  const Params p = P8();  // poly_len = 8
  const std::size_t hamming = 2;

  auto make = [&](std::uint8_t seed) {
    PolyMatrixRaw mat;
    mat.rows = 1;
    mat.cols = 1;
    mat.data.assign(p.poly_len, 0);
    ChaChaRng rng = ChaChaRng::FromSeed(Seed(seed));
    GenTernaryMat(p, mat, hamming, rng);
    return mat;
  };

  const PolyMatrixRaw m = make(42);
  std::size_t ones = 0, negs = 0, zeros = 0;
  for (std::size_t z = 0; z < p.poly_len; ++z) {
    if (m.data[z] == 1)
      ++ones;
    else if (m.data[z] == p.modulus - 1)
      ++negs;
    else if (m.data[z] == 0)
      ++zeros;
    else
      FAIL() << "unexpected coeff " << m.data[z] << " at " << z;
  }
  EXPECT_EQ(ones, hamming);
  EXPECT_EQ(negs, hamming);
  EXPECT_EQ(zeros, p.poly_len - 2 * hamming);

  // Determinism: same seed -> identical secret.
  const PolyMatrixRaw m2 = make(42);
  EXPECT_EQ(m.data, m2.data);
}

}  // namespace
}  // namespace primihub::pir::ypir
