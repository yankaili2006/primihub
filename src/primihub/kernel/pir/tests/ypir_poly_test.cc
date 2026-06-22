/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_poly_test — P2. Round-trips coefficients through HEXL ntt/raw +
 * CRT (raw < modulus is preserved exactly). Built behind tags=[manual]
 * (links @hexl).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"

#include <cstdint>
#include <random>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {
namespace {

Params P(std::size_t poly_len, std::vector<std::uint64_t> moduli) {
  return Params::Init(poly_len, moduli, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                      1, 1, 1, 0, 0);
}

TEST(YpirPolyTest, Shapes) {
  NttContext ctx(P(8, {268369921ull, 249561089ull}));
  EXPECT_EQ(ctx.ZeroRaw(2, 3).data.size(), 2u * 3u * 8u);
  EXPECT_EQ(ctx.ZeroNtt(2, 3).data.size(), 2u * 3u * 2u * 8u);
}

TEST(YpirPolyTest, ToNtt_FromNtt_RoundTrips_TwoModuli) {
  const Params p = P(8, {268369921ull, 249561089ull});
  NttContext ctx(p);
  auto raw = ctx.ZeroRaw(2, 2);
  std::mt19937_64 rng(123);
  for (auto& v : raw.data) v = rng() % p.modulus;
  EXPECT_EQ(ctx.FromNtt(ctx.ToNtt(raw)).data, raw.data);
}

TEST(YpirPolyTest, RoundTrips_SingleModulus) {
  const Params p = P(8, {268369921ull});
  NttContext ctx(p);
  auto raw = ctx.ZeroRaw(1, 2);
  std::mt19937_64 rng(7);
  for (auto& v : raw.data) v = rng() % p.modulus;
  EXPECT_EQ(ctx.FromNtt(ctx.ToNtt(raw)).data, raw.data);
}

}  // namespace
}  // namespace primihub::pir::ypir
