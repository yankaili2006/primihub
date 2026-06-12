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

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_MATRICES_H_
