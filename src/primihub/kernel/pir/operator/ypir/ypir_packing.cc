/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

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

}  // namespace primihub::pir::ypir
