/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_matrices — C++ port of the deterministic, zero-PRNG subset
 * of upstream brave-experiments/frodo-pir@15573960 src/utils.rs
 * `pub mod matrices`. Covers the 3 helpers that move u32 matrices
 * around in row/column form, used by db.rs:
 *
 *   * swap_matrix_fmt          — row<->column transpose
 *   * get_matrix_second_at     — fast single-column extract
 *   * vec_mult_u32_u32         — wrapping u32 dot product
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 2a — sibling of port-order #4. Splits off from chunk 2b
 *   (generate_lwe_matrix_from_seed / random_ternary / random_ternary_vector)
 *   which require a PRNG choice (rand_chacha::ChaCha12Rng-compatible
 *   StdRng::from_seed vs OsRng) and stay deferred.
 *
 * Container choice (the docs/pir/frodo-port-plan.md "decision point"):
 *   Upstream uses Vec<Vec<u32>> end-to-end across utils, db, and api.
 *   We mirror that with std::vector<std::vector<std::uint32_t>>
 *   directly rather than dragging in primihub::pir::core::Matrix:
 *
 *     1. There is no matrix arithmetic in this chunk — the only
 *        operation is row<->column transpose + scalar dot product.
 *        pir_core::Matrix's u32-packed layout buys nothing here.
 *     2. db.rs and api.rs (the next two chunks) keep matrices in
 *        Vec<Vec<u32>> form throughout — switching now would add an
 *        encode/decode pass at every boundary for zero benefit.
 *     3. Byte-for-byte fidelity with upstream test fixtures stays
 *        achievable: the row-major Vec<Vec<u32>> serialises 1:1 to
 *        upstream's serde format. Switching the container would
 *        force us to re-derive every fixture comparison.
 *
 *   We retain the option to swap-in pir_core::Matrix later if a
 *   chunk-7/8 hot path benchmark shows benefit; the helpers in this
 *   header are container-agnostic enough that the refactor would
 *   be local.
 *
 * Field-by-field correspondence with upstream Rust:
 *   pub fn swap_matrix_fmt(matrix: &[Vec<u32>]) -> Vec<Vec<u32>>  →
 *       SwapMatrixFmt(matrix) -> std::vector<std::vector<u32>>
 *   pub fn get_matrix_second_at(matrix: &[Vec<u32>], secidx: usize)
 *       -> Vec<u32>                                                →
 *       GetMatrixSecondAt(matrix, secidx) -> std::vector<u32>
 *   pub fn vec_mult_u32_u32(row, col) -> ResultBoxedError<u32>     →
 *       VecMultU32U32(row, col, *out, *err) -> retcode
 *
 * Bool/error representation: upstream's `ErrorUnexpectedInputSize`
 * maps to `retcode::FAIL` + diagnostic string mentioning the
 * upstream error name (cross-doc traceability), same convention as
 * frodo_format.{h,cc} chunk 1.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_MATRICES_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_MATRICES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_flat_matrix.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

namespace primihub::pir::frodo {

// Row<->column format swap. If `matrix` has shape h x w (h rows of
// w entries each), the result has shape w x h. Upstream assumes
// every row has the same width; we keep that contract — passing a
// jagged matrix is undefined behavior, matching upstream's
// `matrix[0].len()` precondition.
//
// On an empty input (no rows) we return an empty vector — upstream
// would panic on `matrix[0]`. Soft boundary chosen for parity with
// the other ports' "no Rust panic" rule.
std::vector<std::vector<std::uint32_t>> SwapMatrixFmt(
    const std::vector<std::vector<std::uint32_t>>& matrix);

// Returns the column at index `secidx`, i.e. {matrix[0][secidx],
// matrix[1][secidx], ..., matrix[h-1][secidx]}. Equivalent to
// `SwapMatrixFmt(matrix)[secidx]` but with one pass and no
// allocation of the other w-1 columns. Behavior on out-of-range
// `secidx`: returns an empty vector (upstream Rust would panic).
std::vector<std::uint32_t> GetMatrixSecondAt(
    const std::vector<std::vector<std::uint32_t>>& matrix,
    std::size_t secidx);

// Wrapping u32 dot product: `*out = sum_i row[i] *_w col[i]`,
// where `*_w` and `+_w` are u32 wrapping arithmetic (mod 2^32).
// Returns retcode::FAIL with `*err` set if `row.size() !=
// col.size()` (upstream raises ErrorUnexpectedInputSize). On
// success `*err` is left unchanged. `out` must be non-null.
retcode VecMultU32U32(const std::vector<std::uint32_t>& row,
                      const std::vector<std::uint32_t>& col,
                      std::uint32_t* out, std::string* err);



// Generates an LWE matrix A from a public seed by drawing
// `lwe_dim * width` u32s from SeededRng. Mirrors upstream
// `generate_lwe_matrix_from_seed`. Result shape is width rows of
// lwe_dim u32 entries each — i.e. matrix[col][row] indexing,
// matching upstream's Vec<Vec<u32>> column-form.
//
// !! WARNING !! — the underlying PRNG is mt19937_64 in chunk 2b-i.
// See frodo_prng.h for the security regression analysis. Until
// chunk 2b-ii lands the OpenSSL ChaCha20 swap, the LWE matrix A
// generated by this function does NOT satisfy the CSPRNG
// requirement of the FrodoPIR security analysis.
std::vector<std::vector<std::uint32_t>> GenerateLweMatrixFromSeed(
    const SeedBytes& seed, std::size_t lwe_dim, std::size_t width);

// Same algorithm as GenerateLweMatrixFromSeed but writes directly
// into a flat ColMajorMatrix (height=lwe_dim, width=width). Drains
// the SeededRng with ONE FillBytesBulk call over the full
// `width * lwe_dim * 4` byte range -- no per-column allocation
// and no inner-vector page-fault storm.
//
// Byte stream is BIT-EXACT with the per-column overload: column c
// occupies a contiguous run of `lwe_dim` u32s in flat storage at
// offset `c * lwe_dim`, identical to the layout the per-column
// overload writes one column at a time. The
// FrodoMatricesTest GenerateLweMatrixFromSeedFlat_MatchesPerColumn
// _Width2049 test pins this byte-for-byte.
//
// Position in the FrodoPIR port plan (task 7.1):
//   chunk g-1 -- ColMajorMatrix container (frodo_flat_matrix.h)
//   chunk g-2 -- THIS overload. Adds the new entry point. The
//                per-column overload is still the caller of
//                record for BaseParams::GenerateParamsRhs and
//                CommonParams::FromBaseParams -- migration to
//                the flat form lands in chunks g-3..g-5.
ColMajorMatrix GenerateLweMatrixFromSeedFlat(
    const SeedBytes& seed, std::size_t lwe_dim, std::size_t width);



// ====================================================================
// Ternary-distribution sampling (chunk 2c).
//
// Implements the rejection-sampling distribution used by upstream
// FrodoPIR client.gen_query for the LWE secret error vector.
// Mirrors upstream src/utils.rs `pub fn random_ternary` byte-for-
// byte including the slight one-off bias toward the value 0 caused
// by the inclusive lower-bound interval in the trichotomy.
// ====================================================================

// Returns one of {0, 1, UINT32_MAX} via rejection sampling on an
// OsRng-generated u32. Mirrors upstream `random_ternary`. The
// triple {0, 1, UINT32_MAX} represents {0, 1, -1} after mod 2^32
// reduction — what FrodoPIR's LWE secret error vector needs.
//
// Distribution: TERNARY_INTERVAL_SIZE = (UINT32_MAX - 2) / 3.
//   val in [0, TIS]                → 0
//   val in (TIS, 2*TIS]            → 1
//   val in (2*TIS, 3*TIS]          → UINT32_MAX (= -1 mod 2^32)
//   val > 3*TIS                    → reject and resample
// The first interval has TIS+1 integers, the next two have TIS
// each — a 1-out-of-2^32 bias toward 0 that mirrors upstream.
std::uint32_t RandomTernary();

// Width-many independent RandomTernary calls. Mirrors upstream
// `random_ternary_vector(width)`.
std::vector<std::uint32_t> RandomTernaryVector(std::size_t width);

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_MATRICES_H_
