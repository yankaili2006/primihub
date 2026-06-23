/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_regev_test -- random_rng / noise raw constructors. Verifies RNG
 * consumption order + reduction against a parallel ChaChaRng, determinism,
 * and that noise samples come from the DiscreteGaussian.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

#include <algorithm>
#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

Params P8() {
  return Params::Init(8, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                      1, 1, 1, 0, 0);
}

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirRegevTest, RandomRngRaw_ConsumesNextU64PerCoeffReduced) {
  auto p = P8();
  auto rng = ChaChaRng::FromSeed(Seed(7));
  auto rng_ref = ChaChaRng::FromSeed(Seed(7));
  auto a = RandomRngRaw(p, 2, 3, rng);
  EXPECT_EQ(a.rows, 2u);
  EXPECT_EQ(a.cols, 3u);
  ASSERT_EQ(a.data.size(), 2u * 3u * 8u);
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t z = 0; z < 8; ++z) {
        const std::uint64_t expect = rng_ref.NextU64() % p.modulus;
        EXPECT_EQ(a.data[(r * 3 + c) * 8 + z], expect);
      }
  for (auto v : a.data) EXPECT_LT(v, p.modulus);
}

TEST(YpirRegevTest, RandomRngRaw_Deterministic) {
  auto p = P8();
  auto r1 = ChaChaRng::FromSeed(Seed(9));
  auto r2 = ChaChaRng::FromSeed(Seed(9));
  EXPECT_EQ(RandomRngRaw(p, 1, 1, r1).data, RandomRngRaw(p, 1, 1, r2).data);
}

TEST(YpirRegevTest, NoiseRaw_MatchesDiscreteGaussianSamples) {
  auto p = P8();
  const auto dg = DiscreteGaussian::Init(6.4);
  auto rng = ChaChaRng::FromSeed(Seed(3));
  auto rng_ref = ChaChaRng::FromSeed(Seed(3));
  auto e = NoiseRaw(p, 1, 2, dg, rng);
  ASSERT_EQ(e.data.size(), 1u * 2u * 8u);
  for (std::size_t z = 0; z < e.data.size(); ++z) {
    const std::uint64_t sv = rng_ref.NextU64();
    EXPECT_EQ(e.data[z], dg.Sample(p.modulus, sv));
  }
}

TEST(YpirRegevTest, GetRegSample_DecryptsToNoiseExactly) {
  NttContext ctx(P8());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);
  auto sk_rng = ChaChaRng::FromSeed(Seed(1));
  const PolyMatrixRaw sk = RandomRngRaw(p, 1, 1, sk_rng);  // any sk: identity is exact
  auto rng = ChaChaRng::FromSeed(Seed(2));      // noise stream
  auto rng_pub = ChaChaRng::FromSeed(Seed(3));  // uniform-a stream
  const PolyMatrixNTT samp = GetRegSample(ctx, dg, sk, rng, rng_pub);
  ASSERT_EQ(samp.rows, 2u);
  ASSERT_EQ(samp.cols, 1u);
  // reproduce e from a parallel noise rng (same seed/order)
  auto rng_noise_ref = ChaChaRng::FromSeed(Seed(2));
  const PolyMatrixRaw e_ref = NoiseRaw(p, 1, 1, dg, rng_noise_ref);
  // decrypt: phase = p_row0 * sk + p_row1, then FromNtt
  const std::size_t cc_pl = p.crt_count * p.poly_len;
  PolyMatrixNTT row0 = ctx.ZeroNtt(1, 1), row1 = ctx.ZeroNtt(1, 1);
  std::copy(samp.Poly(0, 0, cc_pl), samp.Poly(0, 0, cc_pl) + cc_pl,
            row0.Poly(0, 0, cc_pl));
  std::copy(samp.Poly(1, 0, cc_pl), samp.Poly(1, 0, cc_pl) + cc_pl,
            row1.Poly(0, 0, cc_pl));
  const PolyMatrixNTT sk_ntt = ctx.ToNtt(sk);
  const PolyMatrixNTT phase_ntt = AddNtt(p, MultiplyNtt(p, row0, sk_ntt), row1);
  const PolyMatrixRaw phase = ctx.FromNtt(phase_ntt);
  EXPECT_EQ(phase.data, e_ref.data);  // exact: (-a)*s + (s*a + e) = e
}

TEST(YpirRegevTest, GetFreshRegPublicKey_EachColumnDecryptsToNoise) {
  NttContext ctx(P8());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);
  auto sk_rng = ChaChaRng::FromSeed(Seed(1));
  const PolyMatrixRaw sk = RandomRngRaw(p, 1, 1, sk_rng);
  const std::size_t m = 3;
  auto rng = ChaChaRng::FromSeed(Seed(2));
  auto rng_pub = ChaChaRng::FromSeed(Seed(3));
  const PolyMatrixNTT pk = GetFreshRegPublicKey(ctx, dg, sk, m, rng, rng_pub);
  ASSERT_EQ(pk.rows, 2u);
  ASSERT_EQ(pk.cols, m);
  const PolyMatrixNTT sk_ntt = ctx.ToNtt(sk);
  auto rng_noise_ref = ChaChaRng::FromSeed(Seed(2));  // parallel noise stream
  const std::size_t cc_pl = p.crt_count * p.poly_len;
  for (std::size_t i = 0; i < m; ++i) {
    const PolyMatrixRaw e_ref = NoiseRaw(p, 1, 1, dg, rng_noise_ref);
    PolyMatrixNTT row0 = ctx.ZeroNtt(1, 1), row1 = ctx.ZeroNtt(1, 1);
    std::copy(pk.Poly(0, i, cc_pl), pk.Poly(0, i, cc_pl) + cc_pl,
              row0.Poly(0, 0, cc_pl));
    std::copy(pk.Poly(1, i, cc_pl), pk.Poly(1, i, cc_pl) + cc_pl,
              row1.Poly(0, 0, cc_pl));
    const PolyMatrixNTT phase = AddNtt(p, MultiplyNtt(p, row0, sk_ntt), row1);
    EXPECT_EQ(ctx.FromNtt(phase).data, e_ref.data) << "col " << i;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
