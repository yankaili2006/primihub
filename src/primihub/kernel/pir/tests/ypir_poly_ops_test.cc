/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_poly_ops_test — P3. Verifies NTT-domain matrix add/multiply
 * against a direct reference (pure per-limb modular arithmetic; no HEXL).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"

#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

Params P2x() {  // poly_len=2, crt_count=2
  return Params::Init(2, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                      1, 1, 1, 0, 0);
}

// idx of limb mm coeff z of poly(i,j) in an R x C matrix (poly_len=2,cc=2).
std::size_t Idx(std::size_t i, std::size_t j, std::size_t C, std::size_t mm,
                std::size_t z) {
  return (i * C + j) * 4 + mm * 2 + z;
}

PolyMatrixNTT MakeRand(std::size_t R, std::size_t C, std::mt19937_64& rng) {
  PolyMatrixNTT m;
  m.rows = R; m.cols = C;
  m.data.resize(R * C * 4);
  const std::uint64_t q[2] = {kQ0, kQ1};
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      for (std::size_t mm = 0; mm < 2; ++mm)
        for (std::size_t z = 0; z < 2; ++z)
          m.data[Idx(i, j, C, mm, z)] = rng() % q[mm];
  return m;
}

TEST(YpirPolyOpsTest, Add_MatchesReference) {
  const auto p = P2x();
  std::mt19937_64 rng(11);
  auto a = MakeRand(2, 3, rng), b = MakeRand(2, 3, rng);
  const auto r = AddNtt(p, a, b);
  const std::uint64_t q[2] = {kQ0, kQ1};
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t mm = 0; mm < 2; ++mm)
        for (std::size_t z = 0; z < 2; ++z) {
          const auto e = (a.data[Idx(i,j,3,mm,z)] + b.data[Idx(i,j,3,mm,z)]) % q[mm];
          EXPECT_EQ(r.data[Idx(i,j,3,mm,z)], e);
        }
}

TEST(YpirPolyOpsTest, Multiply_MatchesReference) {
  const auto p = P2x();
  std::mt19937_64 rng(22);
  auto a = MakeRand(2, 3, rng), b = MakeRand(3, 2, rng);  // 2x3 * 3x2
  const auto r = MultiplyNtt(p, a, b);
  ASSERT_EQ(r.rows, 2u);
  ASSERT_EQ(r.cols, 2u);
  const std::uint64_t q[2] = {kQ0, kQ1};
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      for (std::size_t mm = 0; mm < 2; ++mm)
        for (std::size_t z = 0; z < 2; ++z) {
          std::uint64_t acc = 0;
          for (std::size_t k = 0; k < 3; ++k) {
            acc = (acc + (a.data[Idx(i,k,3,mm,z)] * b.data[Idx(k,j,2,mm,z)]) % q[mm]) % q[mm];
          }
          EXPECT_EQ(r.data[Idx(i,j,2,mm,z)], acc) << "i" << i << " j" << j;
        }
}

TEST(YpirPolyOpsTest, PadTopNtt_PrependsZeroRows) {
  auto p = P2x();
  std::mt19937_64 rng(99);
  PolyMatrixNTT a = MakeRand(2, 3, rng);  // 2x3
  PolyMatrixNTT r = PadTopNtt(p, a, 1);   // -> 3x3, top row zero
  EXPECT_EQ(r.rows, 3u);
  EXPECT_EQ(r.cols, 3u);
  for (std::size_t j = 0; j < 3; ++j)
    for (std::size_t mm = 0; mm < 2; ++mm)
      for (std::size_t z = 0; z < 2; ++z)
        EXPECT_EQ(r.data[Idx(0, j, 3, mm, z)], 0u);
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t mm = 0; mm < 2; ++mm)
        for (std::size_t z = 0; z < 2; ++z)
          EXPECT_EQ(r.data[Idx(i + 1, j, 3, mm, z)],
                    a.data[Idx(i, j, 3, mm, z)]);
  EXPECT_EQ(PadTopNtt(p, a, 0).data, a.data);  // pad 0 = identity
}

TEST(YpirPolyOpsTest, CopyIntoNtt_PlacesBlockAtOffset) {
  auto p = P2x();
  std::mt19937_64 rng(123);
  PolyMatrixNTT dst = MakeRand(3, 2, rng);
  for (auto& x : dst.data) x = 0;  // start zeroed
  PolyMatrixNTT src = MakeRand(1, 1, rng);
  CopyIntoNtt(p, dst, src, 2, 1);  // place 1x1 at row2,col1
  for (std::size_t mm = 0; mm < 2; ++mm)
    for (std::size_t z = 0; z < 2; ++z)
      EXPECT_EQ(dst.data[Idx(2, 1, 2, mm, z)], src.data[Idx(0, 0, 1, mm, z)]);
  // everything else still zero
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      if (!(i == 2 && j == 1))
        for (std::size_t mm = 0; mm < 2; ++mm)
          for (std::size_t z = 0; z < 2; ++z)
            EXPECT_EQ(dst.data[Idx(i, j, 2, mm, z)], 0u);
}

TEST(YpirPolyOpsTest, CopyIntoRaw_PlacesBlockAtOffset) {
  auto p = P2x();  // poly_len=2
  PolyMatrixRaw dst; dst.rows = 2; dst.cols = 2; dst.data.assign(2 * 2 * 2, 0);
  PolyMatrixRaw src; src.rows = 1; src.cols = 2; src.data = {11, 12, 13, 14};
  CopyIntoRaw(p, dst, src, 1, 0);  // place 1x2 row at row1
  // dst row1 == src
  EXPECT_EQ(dst.Poly(1, 0, 2)[0], 11u);
  EXPECT_EQ(dst.Poly(1, 0, 2)[1], 12u);
  EXPECT_EQ(dst.Poly(1, 1, 2)[0], 13u);
  EXPECT_EQ(dst.Poly(1, 1, 2)[1], 14u);
  // dst row0 stays zero
  EXPECT_EQ(dst.Poly(0, 0, 2)[0], 0u);
  EXPECT_EQ(dst.Poly(0, 1, 2)[1], 0u);
}

TEST(YpirPolyOpsTest, NegateRaw_MatchesInvertPoly) {
  auto p = P2x();
  PolyMatrixRaw a;
  a.rows = 1; a.cols = 2; a.data = {3, 0, 7, 100};  // poly(0,0)={3,0} (0,1)={7,100}
  auto r = NegateRaw(p, a);
  EXPECT_EQ(r.data[0], p.modulus - 3);
  EXPECT_EQ(r.data[1], p.modulus - 0);  // verbatim: 0 -> modulus, not reduced
  EXPECT_EQ(r.data[2], p.modulus - 7);
  EXPECT_EQ(r.data[3], p.modulus - 100);
}

TEST(YpirPolyOpsTest, NegateNtt_MatchesInvertModular_AndInvolution) {
  auto p = P2x();
  PolyMatrixNTT a;
  a.rows = 1; a.cols = 1; a.data = {5, 0, 9, 0};  // limb0 {5,0} limb1 {9,0}
  auto r = NegateNtt(p, a);
  EXPECT_EQ(r.data[0], (p.moduli[0] - 5) % p.moduli[0]);
  EXPECT_EQ(r.data[1], 0u);  // 0 -> 0
  EXPECT_EQ(r.data[2], (p.moduli[1] - 9) % p.moduli[1]);
  EXPECT_EQ(r.data[3], 0u);
  EXPECT_EQ(NegateNtt(p, r).data, a.data);  // involution
}

TEST(YpirPolyOpsTest, ScalarMultiplyNtt_PointwisePerLimb) {
  auto p = P2x();
  std::mt19937_64 rng(202);
  const PolyMatrixNTT a = MakeRand(1, 1, rng);  // scalar
  const PolyMatrixNTT b = MakeRand(2, 3, rng);
  const auto r = ScalarMultiplyNtt(p, a, b);
  ASSERT_EQ(r.rows, 2u);
  ASSERT_EQ(r.cols, 3u);
  const std::uint64_t q[2] = {kQ0, kQ1};
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t mm = 0; mm < 2; ++mm)
        for (std::size_t z = 0; z < 2; ++z) {
          const std::uint64_t av = a.data[Idx(0, 0, 1, mm, z)];
          const std::uint64_t bv = b.data[Idx(i, j, 3, mm, z)];
          EXPECT_EQ(r.data[Idx(i, j, 3, mm, z)], (av * bv) % q[mm])
              << "i" << i << " j" << j;
        }
}

}  // namespace
}  // namespace primihub::pir::ypir
