/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_poly_ops — P3 of the spiral_rs PolyMatrix port. NTT-domain
 * polynomial-matrix arithmetic (add, multiply) ported from spiral-rs
 * src/poly.rs. These operate on already-transformed PolyMatrixNTT data
 * and are pure modular arithmetic (per-CRT-limb Barrett, via P0) -- no
 * HEXL -- so they are independently testable. The scalar (non-AVX) path
 * is ported; `.50` is Broadwell so the AVX variants are skipped.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_OPS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_OPS_H_

#include <cstddef>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Element-wise matrix add in the NTT domain (a, b same shape). Mirrors
// spiral-rs add(): res[i][j] = a[i][j] + b[i][j] per CRT limb mod q_m.
PolyMatrixNTT AddNtt(const Params& p, const PolyMatrixNTT& a,
                     const PolyMatrixNTT& b);

// Matrix multiply in the NTT domain (a.cols == b.rows). Mirrors
// spiral-rs multiply(): res[i][j] = sum_k a[i][k] (.) b[k][j], where
// (.) is per-CRT-limb pointwise modular multiply and the accumulation
// is modular. Result shape a.rows x b.cols.
PolyMatrixNTT MultiplyNtt(const Params& p, const PolyMatrixNTT& a,
                          const PolyMatrixNTT& b);

// Prepend `pad_rows` zero rows: result is (a.rows + pad_rows) x a.cols,
// with `a` copied into rows [pad_rows, pad_rows + a.rows) and the top
// rows zero. Mirrors spiral-rs PolyMatrix::pad_top.
PolyMatrixNTT PadTopNtt(const Params& p, const PolyMatrixNTT& a,
                        std::size_t pad_rows);

// Copy every poly of `src` into `dst` at block offset (target_row,
// target_col); src must fit. Mirrors spiral-rs PolyMatrix::copy_into.
void CopyIntoRaw(const Params& p, PolyMatrixRaw& dst, const PolyMatrixRaw& src,
                 std::size_t target_row, std::size_t target_col);
void CopyIntoNtt(const Params& p, PolyMatrixNTT& dst, const PolyMatrixNTT& src,
                 std::size_t target_row, std::size_t target_col);

// Negate each coefficient. Raw mirrors spiral invert_poly verbatim
// (modulus - a[i], NOT reduced, so a coeff of 0 maps to modulus). NTT
// mirrors invert_poly_ntt/invert_modular ((moduli[c]-a)%moduli[c], 0->0).
// Together these are -&PolyMatrixRaw / -&PolyMatrixNTT.
PolyMatrixRaw NegateRaw(const Params& p, const PolyMatrixRaw& a);
PolyMatrixNTT NegateNtt(const Params& p, const PolyMatrixNTT& a);

// Scalar (1x1) polynomial multiply in the NTT domain: res[i][j] = a[0][0]
// (.) b[i][j], per-CRT-limb pointwise modular multiply. Mirrors spiral-rs
// scalar_multiply. Result shape == b.
PolyMatrixNTT ScalarMultiplyNtt(const Params& p, const PolyMatrixNTT& a,
                                const PolyMatrixNTT& b);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_OPS_H_
