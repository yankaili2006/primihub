/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

namespace primihub::pir::ypir {

// Faithful port of packing.rs pack_lwes_inner. The Rust uses in-place
// add_into(&mut ct, ...); here AddNtt returns a fresh matrix and we
// reassign, which is functionally identical (no aliasing in this routine).
PolyMatrixNTT PackLwesInner(const NttContext& ctx, std::size_t ell,
                            std::size_t start_idx,
                            const std::vector<PolyMatrixNTT>& rlwe_cts,
                            const std::vector<PolyMatrixNTT>& pub_params,
                            const YConstants& y_constants) {
  const Params& params = ctx.params();
  assert(pub_params.size() == params.poly_len_log2);

  if (ell == 0) {
    return rlwe_cts[start_idx];  // clone (value copy)
  }

  const std::size_t step = static_cast<std::size_t>(1)
                           << (params.poly_len_log2 - ell);
  const std::size_t even = start_idx;
  const std::size_t odd = start_idx + step;

  PolyMatrixNTT ct_even =
      PackLwesInner(ctx, ell - 1, even, rlwe_cts, pub_params, y_constants);
  const PolyMatrixNTT ct_odd =
      PackLwesInner(ctx, ell - 1, odd, rlwe_cts, pub_params, y_constants);

  const PolyMatrixNTT& y = y_constants.y[ell - 1];
  const PolyMatrixNTT& neg_y = y_constants.neg_y[ell - 1];

  const PolyMatrixNTT y_times_ct_odd = ScalarMultiplyNtt(params, y, ct_odd);
  const PolyMatrixNTT neg_y_times_ct_odd =
      ScalarMultiplyNtt(params, neg_y, ct_odd);

  PolyMatrixNTT ct_sum_1 = AddNtt(params, ct_even, neg_y_times_ct_odd);
  ct_even = AddNtt(params, ct_even, y_times_ct_odd);

  const PolyMatrixNTT ct_sum_1_automorphed = HomomorphicAutomorph(
      ctx, (static_cast<std::size_t>(1) << ell) + 1, params.t_exp_left,
      ct_sum_1, pub_params[params.poly_len_log2 - 1 - (ell - 1)]);

  return AddNtt(params, ct_even, ct_sum_1_automorphed);
}

// Faithful port of packing.rs pack_lwes (recursive path). Drives
// PackLwesInner over all poly_len inputs, then injects b_values into the
// constant row in coefficient form, mirroring the upstream out_raw loop:
//   val = barrett_reduction_u128(params, b_values[z] * poly_len)
//   out_raw[1][z] = barrett_u64(params, out_raw[1][z] + val)
PolyMatrixNTT PackLwes(const NttContext& ctx,
                       const std::vector<std::uint64_t>& b_values,
                       const std::vector<PolyMatrixNTT>& rlwe_cts,
                       const std::vector<PolyMatrixNTT>& pub_params,
                       const YConstants& y_constants) {
  const Params& params = ctx.params();
  assert(rlwe_cts.size() == params.poly_len);
  assert(b_values.size() == params.poly_len);

  const PolyMatrixNTT out =
      PackLwesInner(ctx, params.poly_len_log2, 0, rlwe_cts, pub_params,
                    y_constants);
  PolyMatrixRaw out_raw = ctx.FromNtt(out);

  std::uint64_t* row1 = out_raw.Poly(1, 0, params.poly_len);
  for (std::size_t z = 0; z < params.poly_len; ++z) {
    const std::uint64_t val = BarrettReductionU128Raw(
        params.modulus, params.barrett_cr_0_modulus,
        params.barrett_cr_1_modulus,
        static_cast<__uint128_t>(b_values[z]) *
            static_cast<__uint128_t>(params.poly_len));
    row1[z] = BarrettRawU64(row1[z] + val, params.barrett_cr_1_modulus,
                            params.modulus);
  }

  return ctx.ToNtt(out_raw);
}

// Faithful port of packing.rs prep_pack_lwes. Each output column i takes the
// LWE a-vector lwe_cts[j*poly_len + i] (j over poly_len), maps it to RLWE
// row 0 via negacyclic_perm(.,0,modulus), leaves row 1 (b) zero, and NTTs.
std::vector<PolyMatrixNTT> PrepPackLwes(
    const NttContext& ctx, const std::vector<std::uint64_t>& lwe_cts,
    std::size_t cols_to_do) {
  const Params& params = ctx.params();
  assert(lwe_cts.size() == params.poly_len * (params.poly_len + 1));
  assert(cols_to_do == params.poly_len);

  std::vector<PolyMatrixNTT> rlwe_cts;
  rlwe_cts.reserve(cols_to_do);
  for (std::size_t i = 0; i < cols_to_do; ++i) {
    PolyMatrixRaw rlwe_ct = ctx.ZeroRaw(2, 1);
    std::vector<std::uint64_t> poly(params.poly_len);
    for (std::size_t j = 0; j < params.poly_len; ++j)
      poly[j] = lwe_cts[j * params.poly_len + i];
    const std::vector<std::uint64_t> nega =
        NegacyclicPermU64Mod(poly, 0, params.modulus);
    std::uint64_t* row0 = rlwe_ct.Poly(0, 0, params.poly_len);
    for (std::size_t j = 0; j < params.poly_len; ++j) row0[j] = nega[j];
    rlwe_cts.push_back(ctx.ToNtt(rlwe_ct));
  }
  return rlwe_cts;
}

// Faithful port of packing.rs prep_pack_many_lwes. Reshapes the flat
// multi-output buffer (row stride = num_rlwe_outputs*poly_len) into per-output
// (poly_len+1)*poly_len slices, then preps each.
std::vector<std::vector<PolyMatrixNTT>> PrepPackManyLwes(
    const NttContext& ctx, const std::vector<std::uint64_t>& lwe_cts,
    std::size_t num_rlwe_outputs) {
  const Params& params = ctx.params();
  const std::size_t row_stride = num_rlwe_outputs * params.poly_len;
  assert(lwe_cts.size() == (params.poly_len + 1) * row_stride);

  std::vector<std::vector<PolyMatrixNTT>> res;
  res.reserve(num_rlwe_outputs);
  for (std::size_t i = 0; i < num_rlwe_outputs; ++i) {
    std::vector<std::uint64_t> v;
    v.reserve((params.poly_len + 1) * params.poly_len);
    for (std::size_t j = 0; j < params.poly_len + 1; ++j) {
      const std::size_t base = j * row_stride + i * params.poly_len;
      for (std::size_t k = 0; k < params.poly_len; ++k)
        v.push_back(lwe_cts[base + k]);
    }
    res.push_back(PrepPackLwes(ctx, v, params.poly_len));
  }
  return res;
}

// Recursive equivalent of packing.rs pack_many_lwes: one PackLwes per output
// over its poly_len-wide b_values slice (upstream uses precompute; identical
// result).
std::vector<PolyMatrixNTT> PackManyLwes(
    const NttContext& ctx,
    const std::vector<std::vector<PolyMatrixNTT>>& prep_rlwe_cts,
    const std::vector<std::uint64_t>& b_values, std::size_t num_rlwe_outputs,
    const std::vector<PolyMatrixNTT>& pub_params,
    const YConstants& y_constants) {
  const Params& params = ctx.params();
  assert(prep_rlwe_cts.size() == num_rlwe_outputs);
  assert(prep_rlwe_cts[0].size() == params.poly_len);
  assert(b_values.size() == num_rlwe_outputs * params.poly_len);

  std::vector<PolyMatrixNTT> res;
  res.reserve(num_rlwe_outputs);
  for (std::size_t i = 0; i < num_rlwe_outputs; ++i) {
    const std::vector<std::uint64_t> bv(
        b_values.begin() + i * params.poly_len,
        b_values.begin() + (i + 1) * params.poly_len);
    res.push_back(
        PackLwes(ctx, bv, prep_rlwe_cts[i], pub_params, y_constants));
  }
  return res;
}

}  // namespace primihub::pir::ypir
