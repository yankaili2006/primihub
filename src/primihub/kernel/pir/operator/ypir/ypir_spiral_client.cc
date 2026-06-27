/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

namespace primihub::pir::ypir {

std::uint64_t MultiplyUintMod(std::uint64_t a, std::uint64_t b,
                              std::uint64_t modulus) {
  return static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b)) % modulus);
}

std::uint64_t InvertUintMod(std::uint64_t value, std::uint64_t modulus) {
  if (value == 0) return 0;  // None
  // extended_gcd(value, modulus): track only the Bezout coefficient for value.
  std::uint64_t x = value, y = modulus;
  std::int64_t prev_a = 1, a = 0;
  while (y != 0) {
    const std::int64_t q = static_cast<std::int64_t>(x / y);
    const std::uint64_t t = x % y;
    x = y;
    y = t;
    const std::int64_t ta = a;
    a = prev_a - q * a;
    prev_a = ta;
  }
  if (x != 1) return 0;  // gcd != 1 -> None
  if (prev_a < 0)
    return static_cast<std::uint64_t>(prev_a) + modulus;  // u64 wrap == +modulus
  return static_cast<std::uint64_t>(prev_a);
}

PolyMatrixRaw SingleValue(const Params& params, std::uint64_t value) {
  PolyMatrixRaw r;
  r.rows = 1;
  r.cols = 1;
  r.data.assign(params.poly_len, 0);
  r.data[0] = value;
  return r;
}

PolyMatrixRaw MatrixWithIdentity(const Params& params,
                                 const PolyMatrixRaw& in) {
  assert(in.cols == 1);
  const std::size_t rows = in.rows;
  const std::size_t poly_len = params.poly_len;

  PolyMatrixRaw r;
  r.rows = rows;
  r.cols = rows + 1;
  r.data.assign(rows * (rows + 1) * poly_len, 0);

  // column 0 = in
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint64_t* src = in.Poly(row, 0, poly_len);
    std::uint64_t* dst = r.Poly(row, 0, poly_len);
    for (std::size_t z = 0; z < poly_len; ++z) dst[z] = src[z];
  }
  // columns 1..=rows = identity (constant 1 on the diagonal)
  for (std::size_t d = 0; d < rows; ++d) r.Poly(d, 1 + d, poly_len)[0] = 1;

  return r;
}

void GenTernaryMat(const Params& params, PolyMatrixRaw& mat,
                   std::size_t hamming, ChaChaRng& rng) {
  assert(2 * hamming <= params.poly_len);
  const std::uint64_t modulus = params.modulus;
  const std::size_t poly_len = params.poly_len;

  for (std::size_t r = 0; r < mat.rows; ++r) {
    for (std::size_t c = 0; c < mat.cols; ++c) {
      std::uint64_t* pol = mat.Poly(r, c, poly_len);
      for (std::size_t z = 0; z < poly_len; ++z) pol[z] = 0;
      for (std::size_t i = 0; i < hamming; ++i) pol[i] = 1;
      for (std::size_t i = hamming; i < 2 * hamming; ++i) pol[i] = modulus - 1;
      // Fisher-Yates shuffle over the poly_len coefficients.
      for (std::size_t i = poly_len - 1; i >= 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.NextU32()) % (i + 1);
        std::swap(pol[i], pol[j]);
      }
    }
  }
}

namespace {

// spiral client.rs get_scaled_regev_sample: get_regev_sample with the noise e
// scaled by `scale` before b = e_scaled + sk*a. RNG order matches GetRegSample
// (a from rng_pub, e from rng) so the server's public reconstruction lines up.
PolyMatrixNTT ScaledRegevSample(const NttContext& ctx,
                                const DiscreteGaussian& dg,
                                const PolyMatrixRaw& sk_reg, ChaChaRng& rng,
                                ChaChaRng& rng_pub, std::uint64_t scale) {
  const Params& p = ctx.params();
  const PolyMatrixRaw a = RandomRngRaw(p, 1, 1, rng_pub);
  PolyMatrixRaw e = NoiseRaw(p, 1, 1, dg, rng);
  for (std::size_t i = 0; i < p.poly_len; ++i)
    e.data[i] = MultiplyUintMod(e.data[i], scale, p.modulus);
  const PolyMatrixNTT a_ntt = ctx.ToNtt(a);
  const PolyMatrixNTT sk_ntt = ctx.ToNtt(sk_reg);
  const PolyMatrixNTT b_p = MultiplyNtt(p, sk_ntt, a_ntt);
  const PolyMatrixNTT b = AddNtt(p, ctx.ToNtt(e), b_p);
  const PolyMatrixNTT neg_a_ntt = ctx.ToNtt(NegateRaw(p, a));
  PolyMatrixNTT res = ctx.ZeroNtt(2, 1);
  CopyIntoNtt(p, res, neg_a_ntt, 0, 0);
  CopyIntoNtt(p, res, b, 1, 0);
  return res;
}

PolyMatrixNTT FreshScaledRegPublicKey(const NttContext& ctx,
                                      const DiscreteGaussian& dg,
                                      const PolyMatrixRaw& sk_reg, std::size_t m,
                                      ChaChaRng& rng, ChaChaRng& rng_pub,
                                      std::uint64_t scale) {
  const Params& p = ctx.params();
  PolyMatrixNTT res = ctx.ZeroNtt(2, m);
  for (std::size_t i = 0; i < m; ++i) {
    const PolyMatrixNTT s =
        ScaledRegevSample(ctx, dg, sk_reg, rng, rng_pub, scale);
    CopyIntoNtt(p, res, s, 0, i);  // column i
  }
  return res;
}

}  // namespace

Client::Client(const NttContext& ctx, std::size_t hamming, ChaChaRng& key_rng)
    : ctx_(&ctx), dg_(DiscreteGaussian::Init(ctx.params().noise_width)) {
  const Params& p = ctx.params();
  sk_reg_.rows = 1;
  sk_reg_.cols = 1;
  sk_reg_.data.assign(p.poly_len, 0);
  GenTernaryMat(p, sk_reg_, hamming, key_rng);
  sk_reg_full_ = MatrixWithIdentity(p, sk_reg_);  // 1x2
}

PolyMatrixNTT Client::EncryptMatrixReg(const PolyMatrixNTT& a, ChaChaRng& rng,
                                       ChaChaRng& rng_pub) const {
  const Params& p = ctx_->params();
  const PolyMatrixNTT pk =
      GetFreshRegPublicKey(*ctx_, dg_, sk_reg_, a.cols, rng, rng_pub);
  return AddNtt(p, pk, PadTopNtt(p, a, 1));
}

PolyMatrixNTT Client::EncryptMatrixScaledReg(const PolyMatrixNTT& a,
                                             ChaChaRng& rng, ChaChaRng& rng_pub,
                                             std::uint64_t scale) const {
  const Params& p = ctx_->params();
  const PolyMatrixNTT pk =
      FreshScaledRegPublicKey(*ctx_, dg_, sk_reg_, a.cols, rng, rng_pub, scale);
  return AddNtt(p, pk, PadTopNtt(p, a, 1));
}

PolyMatrixNTT Client::DecryptMatrixReg(const PolyMatrixNTT& ct) const {
  const Params& p = ctx_->params();
  return MultiplyNtt(p, ctx_->ToNtt(sk_reg_full_), ct);
}

}  // namespace primihub::pir::ypir
