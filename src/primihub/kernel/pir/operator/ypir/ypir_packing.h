/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_packing -- the recursive RLWE packing from upstream src/packing.rs
 * (task 7.3c, chunk 1). PackLwesInner is the divide-and-conquer butterfly
 * that folds poly_len RLWE ciphertexts (each carrying one value in a
 * coefficient) into a single RLWE ciphertext via Y-constant scalar
 * multiplies + homomorphic automorphisms (HomomorphicAutomorph from
 * ypir_regev). This is the simple, self-contained correctness path; the
 * optimized non-recursive + precompute_pack variant (pack_lwes_inner_non_
 * recursive / precompute_pack / pack_using_precomp_vals) is a later chunk.
 *
 * Deps already ported: HomomorphicAutomorph (ypir_regev), ScalarMultiplyNtt
 * + AddNtt (ypir_poly_ops), YConstants / GenerateYConstants (ypir_server),
 * RawGenerateExpansionParams (ypir_regev, supplies pub_params).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_H_

#include <cstddef>

#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

namespace primihub::pir::ypir {

// Recursive pack butterfly (mirrors packing.rs pack_lwes_inner). Folds the
// poly_len RLWE ciphertexts addressed by (start_idx + k*step) into a single
// 2x1 NTT ciphertext, where step = 1 << (poly_len_log2 - ell). Call with
// ell = params.poly_len_log2 and start_idx = 0 to pack all poly_len inputs.
//
// Preconditions (asserted, mirroring upstream):
//   - pub_params.size() == params.poly_len_log2; each entry is a
//     RawGenerateExpansionParams key-switch key built with
//     m_exp == params.t_exp_left.
//   - rlwe_cts must be indexable up to start_idx + (2^ell - 1)*step; each is
//     a 2x1 NTT ciphertext.
//   - y_constants from GenerateYConstants(ctx).
PolyMatrixNTT PackLwesInner(const NttContext& ctx, std::size_t ell,
                            std::size_t start_idx,
                            const std::vector<PolyMatrixNTT>& rlwe_cts,
                            const std::vector<PolyMatrixNTT>& pub_params,
                            const YConstants& y_constants);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_H_
