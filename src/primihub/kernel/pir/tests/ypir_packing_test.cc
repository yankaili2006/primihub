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

}  // namespace
}  // namespace primihub::pir::ypir
