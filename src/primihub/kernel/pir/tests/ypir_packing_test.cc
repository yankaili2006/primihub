/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_packing_test -- recursive pack butterfly (PackLwesInner, task 7.3c
 * chunk 1). Oracle mirrors ypir_regev_test's PackSingleLwe_FoldEquivalence,
 * generalized to the multi-input divide-and-conquer fold: encrypt poly_len
 * RLWE ciphertexts, pack them homomorphically, decrypt, and compare against
 * the identical butterfly run on the plaintext encodings in R_q (same
 * Y-constants; the homomorphic_automorph step mirrored by plaintext
 * Automorph). P8b (binary expansion gadget) keeps key-switch noise tiny so
 * the fold stays exact through poly_len_log2 levels.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_gadget.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

Params P8b() {  // binary expansion gadget (bits_per=1 -> tiny key-switch
                // noise) so the multi-fold pack stays exact
  return Params::Init(8, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 60, 2, 3, true,
                      1, 1, 1, 0, 0);
}

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirPackingTest, PackLwesInner_RecursiveFoldEquivalence) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);

  auto sk_rng = ChaChaRng::FromSeed(Seed(21));
  const PolyMatrixRaw sk = NoiseRaw(p, 1, 1, dg, sk_rng);  // small secret

  // Expansion key-switch keys: poly_len_log2 of them, m_exp = t_exp_left.
  auto ep_rng = ChaChaRng::FromSeed(Seed(22));
  auto ep_rng_pub = ChaChaRng::FromSeed(Seed(23));
  const std::vector<PolyMatrixNTT> pub_params = RawGenerateExpansionParams(
      ctx, dg, sk, p.poly_len_log2, p.t_exp_left, ep_rng, ep_rng_pub);

  const YConstants yc = GenerateYConstants(ctx);

  // Encrypt poly_len RLWE ciphertexts of distinct plaintext polynomials.
  std::vector<PolyMatrixNTT> rlwe_cts;
  std::vector<PolyMatrixRaw> messages;
  rlwe_cts.reserve(p.poly_len);
  messages.reserve(p.poly_len);
  for (std::size_t k = 0; k < p.poly_len; ++k) {
    PolyMatrixRaw m = ctx.ZeroRaw(1, 1);
    for (std::size_t z = 0; z < p.poly_len; ++z)
      m.data[z] = (z * 29u + 13u + k * 7u) % p.pt_modulus;
    auto enc_rng = ChaChaRng::FromSeed(Seed(static_cast<std::uint8_t>(50 + k)));
    auto enc_rng_pub =
        ChaChaRng::FromSeed(Seed(static_cast<std::uint8_t>(80 + k)));
    rlwe_cts.push_back(RegevEncrypt(ctx, dg, sk, m, enc_rng, enc_rng_pub));
    messages.push_back(m);
  }

  const PolyMatrixNTT packed =
      PackLwesInner(ctx, p.poly_len_log2, 0, rlwe_cts, pub_params, yc);
  const PolyMatrixRaw dec = RegevDecrypt(ctx, sk, packed);

  // Plaintext-domain reference: the same butterfly on the R_q message
  // encodings (Delta * m), Y-constants identical, homomorphic_automorph
  // mirrored by plaintext Automorph.
  std::vector<PolyMatrixNTT> pt_ntt(p.poly_len);
  for (std::size_t k = 0; k < p.poly_len; ++k) {
    PolyMatrixRaw dm = ctx.ZeroRaw(1, 1);
    for (std::size_t z = 0; z < p.poly_len; ++z)
      dm.data[z] = Rescale(messages[k].data[z], p.pt_modulus, p.modulus);
    pt_ntt[k] = ctx.ToNtt(dm);
  }

  std::function<PolyMatrixNTT(std::size_t, std::size_t)> ref_pack =
      [&](std::size_t ell, std::size_t start_idx) -> PolyMatrixNTT {
    if (ell == 0) return pt_ntt[start_idx];
    const std::size_t step = static_cast<std::size_t>(1)
                             << (p.poly_len_log2 - ell);
    PolyMatrixNTT ce = ref_pack(ell - 1, start_idx);
    const PolyMatrixNTT co = ref_pack(ell - 1, start_idx + step);
    const PolyMatrixNTT yco = ScalarMultiplyNtt(p, yc.y[ell - 1], co);
    const PolyMatrixNTT nyco = ScalarMultiplyNtt(p, yc.neg_y[ell - 1], co);
    PolyMatrixNTT cs1 = AddNtt(p, ce, nyco);
    ce = AddNtt(p, ce, yco);
    const std::size_t t = (static_cast<std::size_t>(1) << ell) + 1;
    const PolyMatrixNTT cs1_auto = ctx.ToNtt(Automorph(p, ctx.FromNtt(cs1), t));
    return AddNtt(p, ce, cs1_auto);
  };

  const PolyMatrixRaw ref_raw = ctx.FromNtt(ref_pack(p.poly_len_log2, 0));
  PolyMatrixRaw expected = ctx.ZeroRaw(1, 1);
  for (std::size_t z = 0; z < p.poly_len; ++z)
    expected.data[z] = Rescale(ref_raw.data[z], p.modulus, p.pt_modulus);

  EXPECT_EQ(dec.data, expected.data);
}

TEST(YpirPackingTest, PackLwes_InjectsBValuesIntoConstantRow) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);

  auto sk_rng = ChaChaRng::FromSeed(Seed(21));
  const PolyMatrixRaw sk = NoiseRaw(p, 1, 1, dg, sk_rng);

  auto ep_rng = ChaChaRng::FromSeed(Seed(22));
  auto ep_rng_pub = ChaChaRng::FromSeed(Seed(23));
  const std::vector<PolyMatrixNTT> pub_params = RawGenerateExpansionParams(
      ctx, dg, sk, p.poly_len_log2, p.t_exp_left, ep_rng, ep_rng_pub);

  const YConstants yc = GenerateYConstants(ctx);

  std::vector<PolyMatrixNTT> rlwe_cts;
  std::vector<PolyMatrixRaw> messages;
  rlwe_cts.reserve(p.poly_len);
  messages.reserve(p.poly_len);
  for (std::size_t k = 0; k < p.poly_len; ++k) {
    PolyMatrixRaw m = ctx.ZeroRaw(1, 1);
    for (std::size_t z = 0; z < p.poly_len; ++z)
      m.data[z] = (z * 29u + 13u + k * 7u) % p.pt_modulus;
    auto enc_rng = ChaChaRng::FromSeed(Seed(static_cast<std::uint8_t>(50 + k)));
    auto enc_rng_pub =
        ChaChaRng::FromSeed(Seed(static_cast<std::uint8_t>(80 + k)));
    rlwe_cts.push_back(RegevEncrypt(ctx, dg, sk, m, enc_rng, enc_rng_pub));
    messages.push_back(m);
  }

  // b-values aligned to the rescale lattice (multiples of ~modulus/pt/poly_len)
  // so adding b_values[z]*poly_len shifts each coefficient by whole pt-steps,
  // preserving the same noise margin that makes the fold exact.
  const std::uint64_t base = p.modulus / p.pt_modulus / p.poly_len;
  std::vector<std::uint64_t> b_values(p.poly_len);
  for (std::size_t z = 0; z < p.poly_len; ++z)
    b_values[z] = base * static_cast<std::uint64_t>(z % 7u);

  const PolyMatrixNTT packed =
      PackLwes(ctx, b_values, rlwe_cts, pub_params, yc);
  const PolyMatrixRaw dec = RegevDecrypt(ctx, sk, packed);

  // Reference: identical butterfly on plaintext encodings, then the same
  // Barrett b-value injection into the phase, then rescale back to pt.
  std::vector<PolyMatrixNTT> pt_ntt(p.poly_len);
  for (std::size_t k = 0; k < p.poly_len; ++k) {
    PolyMatrixRaw dm = ctx.ZeroRaw(1, 1);
    for (std::size_t z = 0; z < p.poly_len; ++z)
      dm.data[z] = Rescale(messages[k].data[z], p.pt_modulus, p.modulus);
    pt_ntt[k] = ctx.ToNtt(dm);
  }
  std::function<PolyMatrixNTT(std::size_t, std::size_t)> ref_pack =
      [&](std::size_t ell, std::size_t start_idx) -> PolyMatrixNTT {
    if (ell == 0) return pt_ntt[start_idx];
    const std::size_t step = static_cast<std::size_t>(1)
                             << (p.poly_len_log2 - ell);
    PolyMatrixNTT ce = ref_pack(ell - 1, start_idx);
    const PolyMatrixNTT co = ref_pack(ell - 1, start_idx + step);
    const PolyMatrixNTT yco = ScalarMultiplyNtt(p, yc.y[ell - 1], co);
    const PolyMatrixNTT nyco = ScalarMultiplyNtt(p, yc.neg_y[ell - 1], co);
    PolyMatrixNTT cs1 = AddNtt(p, ce, nyco);
    ce = AddNtt(p, ce, yco);
    const std::size_t t = (static_cast<std::size_t>(1) << ell) + 1;
    const PolyMatrixNTT cs1_auto = ctx.ToNtt(Automorph(p, ctx.FromNtt(cs1), t));
    return AddNtt(p, ce, cs1_auto);
  };
  const PolyMatrixRaw ref_raw = ctx.FromNtt(ref_pack(p.poly_len_log2, 0));

  PolyMatrixRaw expected = ctx.ZeroRaw(1, 1);
  for (std::size_t z = 0; z < p.poly_len; ++z) {
    const std::uint64_t val = BarrettReductionU128Raw(
        p.modulus, p.barrett_cr_0_modulus, p.barrett_cr_1_modulus,
        static_cast<__uint128_t>(b_values[z]) *
            static_cast<__uint128_t>(p.poly_len));
    const std::uint64_t phase = BarrettRawU64(
        ref_raw.data[z] + val, p.barrett_cr_1_modulus, p.modulus);
    expected.data[z] = Rescale(phase, p.modulus, p.pt_modulus);
  }

  EXPECT_EQ(dec.data, expected.data);
}

// prep_pack_lwes: each output column's a-vector lands in RLWE row 0 in
// negacyclic order, row 1 zero. Anchored to NegacyclicPermU64Mod + ToNtt.
TEST(YpirPackingTest, PrepPackLwes_BuildsNegacyclicARows) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  std::vector<std::uint64_t> lwe(p.poly_len * (p.poly_len + 1));
  for (std::size_t idx = 0; idx < lwe.size(); ++idx)
    lwe[idx] = (idx * 1234577u + 7u) % p.modulus;

  const std::vector<PolyMatrixNTT> out = PrepPackLwes(ctx, lwe, p.poly_len);
  ASSERT_EQ(out.size(), p.poly_len);
  for (std::size_t i = 0; i < p.poly_len; ++i) {
    std::vector<std::uint64_t> col(p.poly_len);
    for (std::size_t j = 0; j < p.poly_len; ++j)
      col[j] = lwe[j * p.poly_len + i];
    PolyMatrixRaw expect_raw = ctx.ZeroRaw(2, 1);
    const std::vector<std::uint64_t> nega =
        NegacyclicPermU64Mod(col, 0, p.modulus);
    std::uint64_t* r0 = expect_raw.Poly(0, 0, p.poly_len);
    for (std::size_t j = 0; j < p.poly_len; ++j) r0[j] = nega[j];
    const PolyMatrixNTT expect = ctx.ToNtt(expect_raw);
    EXPECT_EQ(out[i].data, expect.data) << "output column " << i;
  }
}

// prep_pack_many_lwes: per-output reshard equals PrepPackLwes on the manually
// gathered column block.
TEST(YpirPackingTest, PrepPackManyLwes_ReshapesPerOutput) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  const std::size_t num = 3;
  const std::size_t stride = num * p.poly_len;
  std::vector<std::uint64_t> lwe((p.poly_len + 1) * stride);
  for (std::size_t idx = 0; idx < lwe.size(); ++idx)
    lwe[idx] = (idx * 99991u + 17u) % p.modulus;

  const std::vector<std::vector<PolyMatrixNTT>> out =
      PrepPackManyLwes(ctx, lwe, num);
  ASSERT_EQ(out.size(), num);
  for (std::size_t i = 0; i < num; ++i) {
    std::vector<std::uint64_t> v;
    for (std::size_t j = 0; j < p.poly_len + 1; ++j)
      for (std::size_t k = 0; k < p.poly_len; ++k)
        v.push_back(lwe[j * stride + i * p.poly_len + k]);
    const std::vector<PolyMatrixNTT> expect = PrepPackLwes(ctx, v, p.poly_len);
    ASSERT_EQ(out[i].size(), expect.size());
    for (std::size_t c = 0; c < expect.size(); ++c)
      EXPECT_EQ(out[i][c].data, expect[c].data) << "output " << i << " col " << c;
  }
}

// pack_many_lwes: output i equals PackLwes over output i's b_values slice and
// prepared cts (PackLwes itself verified above).
TEST(YpirPackingTest, PackManyLwes_LoopsPackLwes) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);
  auto sk_rng = ChaChaRng::FromSeed(Seed(21));
  const PolyMatrixRaw sk = NoiseRaw(p, 1, 1, dg, sk_rng);
  auto ep_rng = ChaChaRng::FromSeed(Seed(22));
  auto ep_rng_pub = ChaChaRng::FromSeed(Seed(23));
  const std::vector<PolyMatrixNTT> pub_params = RawGenerateExpansionParams(
      ctx, dg, sk, p.poly_len_log2, p.t_exp_left, ep_rng, ep_rng_pub);
  const YConstants yc = GenerateYConstants(ctx);

  const std::size_t num = 2;
  const std::size_t stride = num * p.poly_len;
  std::vector<std::uint64_t> lwe((p.poly_len + 1) * stride);
  for (std::size_t idx = 0; idx < lwe.size(); ++idx)
    lwe[idx] = (idx * 7919u + 3u) % p.modulus;
  const std::vector<std::vector<PolyMatrixNTT>> prep =
      PrepPackManyLwes(ctx, lwe, num);

  std::vector<std::uint64_t> b_values(num * p.poly_len);
  for (std::size_t z = 0; z < b_values.size(); ++z)
    b_values[z] = (z * 131u + 5u) % p.modulus;

  const std::vector<PolyMatrixNTT> many =
      PackManyLwes(ctx, prep, b_values, num, pub_params, yc);
  ASSERT_EQ(many.size(), num);
  for (std::size_t i = 0; i < num; ++i) {
    const std::vector<std::uint64_t> bv(
        b_values.begin() + i * p.poly_len,
        b_values.begin() + (i + 1) * p.poly_len);
    const PolyMatrixNTT ref = PackLwes(ctx, bv, prep[i], pub_params, yc);
    EXPECT_EQ(many[i].data, ref.data) << "output " << i;
  }
}

// The precomputed (fast online) pack must be bit-for-bit identical to the
// recursive PackManyLwes -- it is a pure factoring of the same computation
// (offline butterfly + online b_values injection), so any divergence is a bug.
TEST(YpirPackingTest, PackManyLwesPrecomputed_MatchesRecursive) {
  NttContext ctx(P8b());
  const Params& p = ctx.params();
  const auto dg = DiscreteGaussian::Init(p.noise_width);
  auto sk_rng = ChaChaRng::FromSeed(Seed(21));
  const PolyMatrixRaw sk = NoiseRaw(p, 1, 1, dg, sk_rng);
  auto ep_rng = ChaChaRng::FromSeed(Seed(22));
  auto ep_rng_pub = ChaChaRng::FromSeed(Seed(23));
  const std::vector<PolyMatrixNTT> pub_params = RawGenerateExpansionParams(
      ctx, dg, sk, p.poly_len_log2, p.t_exp_left, ep_rng, ep_rng_pub);
  const YConstants yc = GenerateYConstants(ctx);

  const std::size_t num = 2;
  const std::size_t stride = num * p.poly_len;
  std::vector<std::uint64_t> lwe((p.poly_len + 1) * stride);
  for (std::size_t idx = 0; idx < lwe.size(); ++idx)
    lwe[idx] = (idx * 7919u + 3u) % p.modulus;
  const std::vector<std::vector<PolyMatrixNTT>> prep =
      PrepPackManyLwes(ctx, lwe, num);

  // Precompute the query-independent butterfly once.
  const std::vector<PolyMatrixRaw> precomp =
      PrecomputePackManyLwes(ctx, prep, num, pub_params, yc);
  ASSERT_EQ(precomp.size(), num);

  // Two distinct b_value sets ("queries") reuse the same precomputation.
  for (std::uint64_t salt : {5u, 9973u}) {
    std::vector<std::uint64_t> b_values(num * p.poly_len);
    for (std::size_t z = 0; z < b_values.size(); ++z)
      b_values[z] = (z * 131u + salt) % p.modulus;

    const std::vector<PolyMatrixNTT> recursive =
        PackManyLwes(ctx, prep, b_values, num, pub_params, yc);
    const std::vector<PolyMatrixNTT> fast =
        PackManyLwesPrecomputed(ctx, precomp, b_values, num);
    ASSERT_EQ(fast.size(), num);
    for (std::size_t i = 0; i < num; ++i)
      EXPECT_EQ(fast[i].data, recursive[i].data)
          << "salt=" << salt << " output " << i;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
