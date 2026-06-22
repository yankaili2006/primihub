/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_packing_fast — the HEXL-free / non-AVX helper ops from upstream
 * src/packing.rs (chunk 9). The big packing algorithm (pack_lwes_*,
 * precompute_pack, ...) and the AVX512 kernels (fast_multiply_no_reduce,
 * multiply_*_avx) are deferred (the latter need AVX512, absent on the
 * Broadwell build host); this lands the bounded, independently-verifiable
 * pieces: the condensed CRT representation (two ~28-bit NTT limbs packed
 * into one u64) and the lazy single-word Barrett (no final correction).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_FAST_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_FAST_H_

#include <cstdint>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Lazy single-word Barrett: input - ((input*cr1)>>64)*modulus, WITHOUT
// the final conditional subtraction (so the result is congruent to
// input mod modulus but may lie in [0, 2*modulus)). Mirrors
// fast_barrett_raw_u64; used inside the lazy-reduction packing pipeline.
std::uint64_t FastBarrettRawU64(std::uint64_t input, std::uint64_t cr1,
                                std::uint64_t modulus);

// Pack the two CRT limbs of each NTT coefficient into the low/high 32
// bits of one u64: res[z] = a[z] | (a[z+poly_len] << 32). The condensed
// matrix uses only the first poly_len slots per poly. Mirrors
// condense_matrix. Requires each limb < 2^32 (true for the ~2^28 moduli).
PolyMatrixNTT CondenseMatrix(const Params& p, const PolyMatrixNTT& a);

// Inverse of CondenseMatrix: split each packed u64 back into two limbs.
// Mirrors uncondense_matrix.
PolyMatrixNTT UncondenseMatrix(const Params& p, const PolyMatrixNTT& a);

// Scalar port of fast_multiply_no_reduce (the upstream version is AVX512,
// absent on the Broadwell host; this computes the identical result). `a`
// is 1xK and `b` is Kx1, both CONDENSED; multiplies per CRT limb and
// accumulates WITHOUT modular reduction (the lo/hi 32-bit halves are the
// two limbs). Returns a 1x1 UNCONDENSED matrix (limb m at [m*poly_len..]).
// Caller must FastReduce before the values are used modularly; valid while
// the un-reduced accumulators stay below 2^64 (K * (2^28)^2).
PolyMatrixNTT FastMultiplyNoReduce(const Params& p, const PolyMatrixNTT& a,
                                   const PolyMatrixNTT& b);

// Reduce every limb of `res` mod its CRT modulus (barrett_coeff_u64).
// Mirrors fast_reduce. Closes a lazy-reduction sequence.
void FastReduce(const Params& p, PolyMatrixNTT& res);

// res[i] += a[i] then lazy single-word Barrett (mirrors fast_add_into;
// result congruent mod q, possibly in [0, 2q)).
void FastAddInto(const Params& p, PolyMatrixNTT& res, const PolyMatrixNTT& a);

// res[i] += a[i] with no reduction at all (mirrors fast_add_into_no_reduce).
void FastAddIntoNoReduce(PolyMatrixNTT& res, const PolyMatrixNTT& a);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PACKING_FAST_H_
