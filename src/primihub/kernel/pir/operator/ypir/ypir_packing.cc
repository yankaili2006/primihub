/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"

#include <cassert>
#include <cstddef>

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

}  // namespace primihub::pir::ypir
