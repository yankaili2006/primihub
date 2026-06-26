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
#include <cstdint>

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

// Full pack (mirrors packing.rs pack_lwes, recursive path): folds the
// poly_len RLWE ciphertexts via PackLwesInner, then injects the per-output
// LWE b-values into the constant (row 1) polynomial -- coefficient z gets
// += b_values[z] * poly_len (mod modulus, Barrett-reduced), matching the
// upstream out_raw normalization. Returns a 2x1 NTT ciphertext.
//
// (Upstream pack_lwes drives pack_lwes_inner_non_recursive with an optional
// precomputed-vals fast path; this recursive form produces the identical
// result -- the precompute optimization + b_values fast path is a later
// chunk.)
//
// Preconditions: rlwe_cts.size() == params.poly_len; b_values.size() ==
// params.poly_len; pub_params/y_constants as for PackLwesInner.
PolyMatrixNTT PackLwes(const NttContext& ctx,
                       const std::vector<std::uint64_t>& b_values,
                       const std::vector<PolyMatrixNTT>& rlwe_cts,
                       const std::vector<PolyMatrixNTT>& pub_params,
                       const YConstants& y_constants);

// Build the poly_len "prepared" RLWE ciphertexts for one packed output from
// a flat LWE buffer (mirrors packing.rs prep_pack_lwes). lwe_cts holds
// poly_len*(poly_len+1) u64 as (poly_len+1) rows of poly_len; for column i
// the a-vector (rows 0..poly_len-1) becomes RLWE row 0 in negacyclic order
// (NegacyclicPermU64Mod shift 0), row 1 (the b) left zero -- b's are injected
// separately by PackLwes. cols_to_do must equal params.poly_len. Returns
// poly_len 2x1 NTT ciphertexts.
std::vector<PolyMatrixNTT> PrepPackLwes(
    const NttContext& ctx, const std::vector<std::uint64_t>& lwe_cts,
    std::size_t cols_to_do);

// Reshape a flat multi-output LWE buffer and prep each output (mirrors
// packing.rs prep_pack_many_lwes). lwe_cts holds (poly_len+1) *
// (num_rlwe_outputs*poly_len) u64; output i gathers its poly_len-wide column
// block across all poly_len+1 rows, then PrepPackLwes. Returns
// num_rlwe_outputs vectors of poly_len ciphertexts each.
std::vector<std::vector<PolyMatrixNTT>> PrepPackManyLwes(
    const NttContext& ctx, const std::vector<std::uint64_t>& lwe_cts,
    std::size_t num_rlwe_outputs);

// Pack many RLWE outputs (recursive path; mirrors packing.rs pack_many_lwes).
// For output i: PackLwes(b_values[i*poly_len .. (i+1)*poly_len],
// prep_rlwe_cts[i], pub_params, y_constants). Upstream drives a precomputed
// (Precomp) fast path; this loops the recursive PackLwes for the identical
// result (the precompute optimization is a later chunk). b_values.size() ==
// num_rlwe_outputs*poly_len. Returns num_rlwe_outputs packed 2x1 NTT cts.
std::vector<PolyMatrixNTT> PackManyLwes(
    const NttContext& ctx,
    const std::vector<std::vector<PolyMatrixNTT>>& prep_rlwe_cts,
    const std::vector<std::uint64_t>& b_values, std::size_t num_rlwe_outputs,
    const std::vector<PolyMatrixNTT>& pub_params,
    const YConstants& y_constants);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_H_
