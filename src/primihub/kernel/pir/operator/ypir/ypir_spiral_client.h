/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_spiral_client — the Regev (RLWE) encryption surface of spiral_rs
 * `Client` that YPIR query generation depends on (see
 * docs/pir/spiral-client-port-plan.md). This file is built up across
 * sub-chunks:
 *   - 12b-1 (this commit): the arith/helper gaps — InvertUintMod /
 *     MultiplyUintMod / SingleValue / MatrixWithIdentity / GenTernaryMat
 *     (+ kHammingWeight).
 *   - 12b-2: the Client class (ternary sk_reg, scaled samples,
 *     EncryptMatrix{,Scaled}Reg, DecryptMatrixReg).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SPIRAL_CLIENT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SPIRAL_CLIENT_H_

#include <cstddef>
#include <cstdint>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// spiral_rs client.rs HAMMING_WEIGHT — ternary secret has this many +1s and
// this many -1s (the rest zero).
inline constexpr std::size_t kHammingWeight = 256;

// number_theory.rs multiply_uint_mod: (a*b) mod modulus via u128 (no Barrett).
std::uint64_t MultiplyUintMod(std::uint64_t a, std::uint64_t b,
                              std::uint64_t modulus);

// number_theory.rs invert_uint_mod via extended_gcd. Returns the modular
// inverse of value mod modulus, or 0 when none exists (value == 0 or
// gcd(value, modulus) != 1 — upstream returns Option::None there; 0 is never a
// valid inverse so it is a safe sentinel).
std::uint64_t InvertUintMod(std::uint64_t value, std::uint64_t modulus);

// poly.rs PolyMatrixRaw::single_value: a 1x1 raw polynomial whose constant
// coefficient is `value` (all higher coefficients 0).
PolyMatrixRaw SingleValue(const Params& params, std::uint64_t value);

// client.rs matrix_with_identity: given a rows x 1 raw matrix `in`, returns the
// rows x (rows+1) matrix [in | I_rows] (column 0 is `in`, the remaining rows
// columns form the identity: constant-1 on the diagonal).
PolyMatrixRaw MatrixWithIdentity(const Params& params, const PolyMatrixRaw& in);

// client.rs gen_ternary_mat: fill every polynomial of `mat` with `hamming` +1
// coefficients and `hamming` (modulus-1) coefficients (rest 0), then shuffle
// each polynomial with a Fisher-Yates pass driven by rng. Requires
// 2*hamming <= params.poly_len and `mat` already sized
// rows*cols*poly_len. (YPIR is single-impl, so the shuffle need only be a
// valid, deterministic permutation -- not byte-identical to rand's shuffle.)
void GenTernaryMat(const Params& params, PolyMatrixRaw& mat, std::size_t hamming,
                   ChaChaRng& rng);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SPIRAL_CLIENT_H_
