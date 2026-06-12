/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_database — C++ port of upstream brave-experiments/
 * frodo-pir@15573960 src/db.rs `pub struct Database` and the
 * file-private `construct_rows` helper. Holds the DB matrix in
 * Vec<Vec<u32>> form (m rows of `get_matrix_width(elem_size,
 * plaintext_bits)` u32 entries each) plus three immutable size
 * parameters, and exposes:
 *
 *   * Database(entries, m, elem_size, plaintext_bits)
 *                         — chunk 3a; data ctor from pre-decoded
 *                           entries (column-form by convention)
 *   * SwitchFmt           — chunk 3a; calls SwapMatrixFmt on the
 *                           entries
 *   * VecMult             — chunk 3a; wrapping dot-product of
 *                           an input row against the col_idx-th
 *                           DB column
 *   * GetRow              — chunk 3a; returns row i (clone)
 *   * GetMatrixWidth      — chunk 3a; static; rounds up
 *                           elem_size / plaintext_bits
 *   * GetMatrixWidthSelf  — chunk 3a; reads its own params
 *   * GetMatrixHeight     — chunk 3a; returns m
 *   * GetElemSize         — chunk 3a; returns elem_size
 *   * GetPlaintextBits    — chunk 3a; returns plaintext_bits
 *   * ConstructRows       — chunk 3b; free fn; base64 strings
 *                           → row-form Vec<Vec<u32>>
 *   * Database::New       — chunk 3b; static factory from
 *                           base64-encoded strings, column-form
 *                           result
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunks 3a + 3b — port-order #5 (db.rs, 260 LOC). 3c will
 *   still cover from_file / write_to_file / get_db_entry.
 *
 * Construction paths:
 *   * Direct data ctor (chunk 3a): pre-decoded Vec<Vec<u32>> in
 *     column-form. Tests use this to skip base64.
 *   * Database::New (chunk 3b): m base64-encoded strings →
 *     decode → bits → u32 chunks of plaintext_bits each →
 *     transpose to column-form → Database. Mirrors upstream
 *     `Database::new(elements, m, elem_size, plaintext_bits)`.
 *
 * Field-by-field correspondence with upstream Rust:
 *   pub struct Database { entries, m, elem_size, plaintext_bits } →
 *       class Database with the same four fields, immutable params
 *       and mutable entries (so SwitchFmt can mutate in-place).
 *   pub fn switch_fmt(&mut self)                                   →
 *       SwitchFmt() (in-place via SwapMatrixFmt)
 *   pub fn vec_mult(&self, row, col_idx) -> u32                    →
 *       VecMult(row, col_idx) -> uint32_t (panics on OOB col_idx;
 *       we keep upstream's `.unwrap()` semantics — caller must
 *       satisfy `col_idx < entries.size()`)
 *   pub fn get_row(&self, i) -> Vec<u32>                           →
 *       GetRow(i) (returns empty vec on OOB, soft boundary)
 *   pub fn get_matrix_width(elem_size, plaintext_bits) -> usize    →
 *       static GetMatrixWidth(elem_size, plaintext_bits)
 *   pub fn get_matrix_width_self(&self) -> usize                   →
 *       GetMatrixWidthSelf() const
 *   pub fn get_matrix_height(&self) -> usize                       →
 *       GetMatrixHeight() const { return m; }
 *   pub fn get_elem_size(&self) -> usize                           →
 *       GetElemSize() const
 *   pub fn get_plaintext_bits(&self) -> usize                      →
 *       GetPlaintextBits() const
 *   fn construct_rows(elements, m, elem_size, plaintext_bits)
 *       -> ResultBoxedError<Vec<Vec<u32>>>                         →
 *       free fn ConstructRows(elements, m, elem_size,
 *       plaintext_bits, *out, *err) -> retcode (file-private in
 *       upstream; we expose it free-function in the namespace
 *       for direct testability)
 *   pub fn new(elements, m, elem_size, plaintext_bits)
 *       -> ResultBoxedError<Self>                                  →
 *       static Database::New(elements, m, elem_size,
 *       plaintext_bits, *out, *err) -> retcode
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"

namespace primihub::pir::frodo {

class Database {
 public:
  // Construct from already-decoded entries. `entries` is the DB
  // matrix in the layout produced by SwapMatrixFmt of the row-form
  // produced by ConstructRows (chunk 3b) — i.e. column-form by
  // default; SwitchFmt() toggles between forms in-place. The
  // ctor does NOT validate that entries has shape m x
  // get_matrix_width(elem_size, plaintext_bits) — upstream Rust
  // doesn't either; the invariant is established by the caller
  // (Database::New, below).
  Database(std::vector<std::vector<std::uint32_t>> entries,
           std::size_t m, std::size_t elem_size,
           std::size_t plaintext_bits);

  // Static factory mirroring upstream `Database::new`. Decodes each
  // of `elements` from base64 to bytes, chunks the bit-stream into
  // u32 entries of `plaintext_bits` width via the chunk-1 format
  // helpers, then transposes the m row-form result into column-form.
  // Returns retcode::FAIL with `*err` set on:
  //   * elements.size() != m (size mismatch)
  //   * base64 decode failure (any element decodes to empty when the
  //     input was non-empty — primihub base64_decode returns "")
  //   * BitsToU32Le size overflow (chunk-1 propagation)
  // On success `*out_db` receives the constructed Database; `*err`
  // is left unchanged.
  static retcode New(const std::vector<std::string>& elements,
                     std::size_t m, std::size_t elem_size,
                     std::size_t plaintext_bits, Database* out_db,
                     std::string* err);

  // In-place row<->column format swap. Calls SwapMatrixFmt on the
  // entries vector and assigns the result back.
  void SwitchFmt();

  // Wrapping u32 dot product of `row` against the col_idx-th
  // column of `entries`. Caller MUST satisfy col_idx <
  // entries.size() and row.size() == entries[col_idx].size() —
  // upstream uses `.unwrap()` so any size mismatch would panic.
  // We mirror that: an invalid index aborts via assertion in
  // debug builds; in release we return 0 to avoid UB (NOT the
  // upstream behavior, but a soft boundary matching the rest
  // of the FrodoPIR port).
  std::uint32_t VecMult(const std::vector<std::uint32_t>& row,
                        std::size_t col_idx) const;

  // Returns a copy of the i-th row of the entries matrix. On OOB
  // returns an empty vector (upstream Rust would panic).
  std::vector<std::uint32_t> GetRow(std::size_t i) const;

  // Returns the i-th DB entry as a base64-encoded string. Mirrors
  // upstream `get_db_entry(&self, i: usize) -> String` =
  // base64_from_u32_slice(get_matrix_second_at(entries, i),
  //                       plaintext_bits, elem_size).
  // On OOB returns empty string (upstream Rust would panic on
  // get_matrix_second_at indexing).
  std::string GetDbEntry(std::size_t i) const;

  // Static helper: width of the DB matrix = ceil(elem_size /
  // plaintext_bits). Returns 0 if plaintext_bits is 0 (upstream
  // would panic on integer-divide-by-zero).
  static std::size_t GetMatrixWidth(std::size_t elem_size,
                                    std::size_t plaintext_bits);

  // Same as GetMatrixWidth but reads this->elem_size /
  // this->plaintext_bits.
  std::size_t GetMatrixWidthSelf() const;
  std::size_t GetMatrixHeight() const { return m_; }
  std::size_t GetElemSize() const { return elem_size_; }
  std::size_t GetPlaintextBits() const { return plaintext_bits_; }

  // Test-only accessor exposing the underlying matrix for
  // hand-computed roundtrip checks. Not part of the upstream API.
  const std::vector<std::vector<std::uint32_t>>& EntriesForTest() const {
    return entries_;
  }

  // Default-constructible so callers can declare a placeholder
  // before Database::New writes into it via *out_db. The default
  // state is an empty 0×0 DB with m=elem_size=plaintext_bits=0; do
  // not use it without subsequently calling Database::New or
  // overwriting via the data ctor.
  Database();

 private:
  std::vector<std::vector<std::uint32_t>> entries_;
  std::size_t m_;
  std::size_t elem_size_;
  std::size_t plaintext_bits_;
};

// Free-function helper that mirrors upstream's file-private
// `construct_rows`. For each of the `m` base64-encoded strings in
// `elements`, decodes to bytes, bit-decomposes LSB-first
// (BytesToBitsLe), then chunks the bit-stream into row_width =
// GetMatrixWidth(elem_size, plaintext_bits) u32 entries of
// `plaintext_bits` width (BitsToU32Le for each chunk; the last
// chunk uses whatever bits remain). Returns retcode::FAIL on:
//   * elements.size() != m (size mismatch)
//   * any chunk-1 BitsToU32Le failure (e.g. plaintext_bits > 32)
// On success `*out` receives the m-rows-by-row_width matrix.
//
// Exposed (vs. file-private in upstream) for testability — the
// chunk-3b tests hand-compute single-byte fixtures against this
// directly without dragging Database in.
retcode ConstructRows(const std::vector<std::string>& elements,
                      std::size_t m, std::size_t elem_size,
                      std::size_t plaintext_bits,
                      std::vector<std::vector<std::uint32_t>>* out,
                      std::string* err);

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_
