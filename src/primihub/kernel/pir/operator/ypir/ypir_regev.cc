/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"

namespace primihub::pir::ypir {

PolyMatrixRaw RandomRngRaw(const Params& p, std::size_t rows, std::size_t cols,
                           ChaChaRng& rng) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = rows;
  out.cols = cols;
  out.data.assign(rows * cols * pl, 0);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      std::uint64_t* poly = out.Poly(r, c, pl);
      for (std::size_t z = 0; z < pl; ++z) poly[z] = rng.NextU64() % p.modulus;
    }
  }
  return out;
}

PolyMatrixRaw NoiseRaw(const Params& p, std::size_t rows, std::size_t cols,
                       const DiscreteGaussian& dg, ChaChaRng& rng) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = rows;
  out.cols = cols;
  out.data.assign(rows * cols * pl, 0);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      std::uint64_t* poly = out.Poly(r, c, pl);
      for (std::size_t z = 0; z < pl; ++z) {
        const std::uint64_t sampled_val = rng.NextU64();
        poly[z] = dg.Sample(p.modulus, sampled_val);
      }
    }
  }
  return out;
}

PolyMatrixNTT GetRegSample(const NttContext& ctx, const DiscreteGaussian& dg,
                           const PolyMatrixRaw& sk_reg, ChaChaRng& rng,
                           ChaChaRng& rng_pub) {
  const Params& p = ctx.params();
  const PolyMatrixRaw a = RandomRngRaw(p, 1, 1, rng_pub);
  const PolyMatrixRaw e = NoiseRaw(p, 1, 1, dg, rng);
  const PolyMatrixNTT a_ntt = ctx.ToNtt(a);
  const PolyMatrixNTT sk_ntt = ctx.ToNtt(sk_reg);
  const PolyMatrixNTT b_p = MultiplyNtt(p, sk_ntt, a_ntt);  // sk*a
  const PolyMatrixNTT b = AddNtt(p, ctx.ToNtt(e), b_p);     // e + sk*a
  const PolyMatrixNTT neg_a_ntt = ctx.ToNtt(NegateRaw(p, a));
  PolyMatrixNTT res = ctx.ZeroNtt(2, 1);
  CopyIntoNtt(p, res, neg_a_ntt, 0, 0);
  CopyIntoNtt(p, res, b, 1, 0);
  return res;
}

PolyMatrixNTT GetFreshRegPublicKey(const NttContext& ctx,
                                   const DiscreteGaussian& dg,
                                   const PolyMatrixRaw& sk_reg, std::size_t m,
                                   ChaChaRng& rng, ChaChaRng& rng_pub) {
  const Params& p = ctx.params();
  PolyMatrixNTT res = ctx.ZeroNtt(2, m);
  for (std::size_t i = 0; i < m; ++i) {
    const PolyMatrixNTT sample = GetRegSample(ctx, dg, sk_reg, rng, rng_pub);
    CopyIntoNtt(p, res, sample, 0, i);  // column i
  }
  return res;
}

}  // namespace primihub::pir::ypir
