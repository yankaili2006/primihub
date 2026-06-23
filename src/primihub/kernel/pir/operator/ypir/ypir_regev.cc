/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

#include <algorithm>

#include "src/primihub/kernel/pir/operator/ypir/ypir_gadget.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
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

std::vector<PolyMatrixNTT> RawGenerateExpansionParams(
    const NttContext& ctx, const DiscreteGaussian& dg,
    const PolyMatrixRaw& sk_reg, std::size_t num_exp, std::size_t m_exp,
    ChaChaRng& rng, ChaChaRng& rng_pub) {
  const Params& p = ctx.params();
  const PolyMatrixNTT g_exp_ntt = ctx.ToNtt(BuildGadget(p, 1, m_exp));
  std::vector<PolyMatrixNTT> res;
  res.reserve(num_exp);
  for (std::size_t i = 0; i < num_exp; ++i) {
    const std::size_t t = (p.poly_len >> i) + 1;
    const PolyMatrixRaw tau_sk_reg = Automorph(p, sk_reg, t);
    const PolyMatrixNTT prod = MultiplyNtt(p, ctx.ToNtt(tau_sk_reg), g_exp_ntt);
    const PolyMatrixNTT sample =
        GetFreshRegPublicKey(ctx, dg, sk_reg, m_exp, rng, rng_pub);
    res.push_back(AddNtt(p, sample, PadTopNtt(p, prod, 1)));
  }
  return res;
}

PolyMatrixNTT RegevEncrypt(const NttContext& ctx, const DiscreteGaussian& dg,
                           const PolyMatrixRaw& sk_reg, const PolyMatrixRaw& m,
                           ChaChaRng& rng, ChaChaRng& rng_pub) {
  const Params& p = ctx.params();
  const std::size_t pl = p.poly_len;
  const PolyMatrixNTT ct = GetRegSample(ctx, dg, sk_reg, rng, rng_pub);
  PolyMatrixRaw dm = ctx.ZeroRaw(2, 1);  // row0 = 0, row1 = Delta*m
  std::uint64_t* dm_row1 = dm.Poly(1, 0, pl);
  const std::uint64_t* mp = m.Poly(0, 0, pl);
  for (std::size_t z = 0; z < pl; ++z)
    dm_row1[z] = Rescale(mp[z], p.pt_modulus, p.modulus);
  return AddNtt(p, ct, ctx.ToNtt(dm));
}

PolyMatrixRaw RegevDecrypt(const NttContext& ctx, const PolyMatrixRaw& sk_reg,
                           const PolyMatrixNTT& ct) {
  const Params& p = ctx.params();
  const std::size_t pl = p.poly_len;
  const std::size_t cc_pl = p.crt_count * pl;
  PolyMatrixNTT row0 = ctx.ZeroNtt(1, 1), row1 = ctx.ZeroNtt(1, 1);
  std::copy(ct.Poly(0, 0, cc_pl), ct.Poly(0, 0, cc_pl) + cc_pl,
            row0.Poly(0, 0, cc_pl));
  std::copy(ct.Poly(1, 0, cc_pl), ct.Poly(1, 0, cc_pl) + cc_pl,
            row1.Poly(0, 0, cc_pl));
  const PolyMatrixNTT sk_ntt = ctx.ToNtt(sk_reg);
  const PolyMatrixNTT phase_ntt = AddNtt(p, MultiplyNtt(p, row0, sk_ntt), row1);
  const PolyMatrixRaw phase = ctx.FromNtt(phase_ntt);
  PolyMatrixRaw out = ctx.ZeroRaw(1, 1);
  const std::uint64_t* php = phase.Poly(0, 0, pl);
  std::uint64_t* op = out.Poly(0, 0, pl);
  for (std::size_t z = 0; z < pl; ++z)
    op[z] = Rescale(php[z], p.modulus, p.pt_modulus);
  return out;
}

PolyMatrixNTT HomomorphicAutomorph(const NttContext& ctx, std::size_t t,
                                   std::size_t t_exp, const PolyMatrixNTT& ct,
                                   const PolyMatrixNTT& pub_param) {
  const Params& p = ctx.params();
  const std::size_t pl = p.poly_len;
  // ct (2x1) -> raw -> Galois automorph
  const PolyMatrixRaw ct_raw = ctx.FromNtt(ct);
  const PolyMatrixRaw ct_auto = Automorph(p, ct_raw, t);  // 2x1 raw
  // gadget-decompose row 0 of ct_auto into t_exp digits (rdim=1)
  PolyMatrixRaw ginv_ct = GadgetInvertRdim(p, t_exp, ct_auto, 1);  // t_exp x 1
  // upstream transforms only rows 1..t_exp (drops the 2^0 digit). Equivalent:
  // zero row 0 then ToNtt the whole matrix (NTT linear, NTT(0)=0).
  std::uint64_t* g0 = ginv_ct.Poly(0, 0, pl);
  for (std::size_t z = 0; z < pl; ++z) g0[z] = 0;
  const PolyMatrixNTT ginv_ct_ntt = ctx.ToNtt(ginv_ct);
  // key-switch: pub_param (2 x t_exp) * ginv_ct_ntt (t_exp x 1) -> 2x1
  const PolyMatrixNTT w_times = MultiplyNtt(p, pub_param, ginv_ct_ntt);
  // row 1 of ct_auto -> 1x1 -> NTT, padded to row 1, plus the key-switch term
  PolyMatrixRaw ct_auto_1 = ctx.ZeroRaw(1, 1);
  std::copy(ct_auto.Poly(1, 0, pl), ct_auto.Poly(1, 0, pl) + pl,
            ct_auto_1.Poly(0, 0, pl));
  const PolyMatrixNTT ct_auto_1_ntt = ctx.ToNtt(ct_auto_1);
  return AddNtt(p, PadTopNtt(p, ct_auto_1_ntt, 1), w_times);
}

}  // namespace primihub::pir::ypir
