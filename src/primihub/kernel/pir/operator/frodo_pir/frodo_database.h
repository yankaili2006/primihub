/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_database — C++ port of the deterministic, base64-free
 * subset of upstream brave-experiments/frodo-pir@15573960
 * src/db.rs `pub struct Database`. Holds the DB matrix in
 * Vec<Vec<u32>> form (m rows of `get_matrix_width(elem_size,
 * plaintext_bits)` u32 entries each) plus three immutable size
 * parameters, and exposes:
 *
 *   * SwitchFmt           — calls SwapMatrixFmt on the entries
 *   * VecMult             — wrapping dot-product of an input row
 *                           against the col_idx-th DB column
 *   * GetRow              — returns row i (clone)
 *   * GetMatrixWidth      — static; rounds up elem_size /
 *                           plaintext_bits
 *   * GetMatrixWidthSelf  — same but reads its own params
 *   * GetMatrixHeight     — returns m
 *   * GetElemSize         — returns elem_size
 *   * GetPlaintextBits    — returns plaintext_bits
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 3a — first sub-chunk of port-order #5 (db.rs, 260 LOC).
 *   Covers the data container + simple accessors. Splits off from
 *   chunks 3b (ConstructRows + base64-string ctor) and 3c
 *   (from_file / write_to_file / get_db_entry).
 *
 * Construction:
 *   The ctor takes already-decoded `Vec<Vec<u32>>` entries plus
 *   the three immutable params. The base64-string ctor that
 *   upstream calls `new` lives in chunk 3b alongside the helper
 *   `ConstructRows`. Tests can construct DBs directly via the
 *   data ctor without dragging base64 in.
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
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

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
  // (chunk 3b's Database::New).
  Database(std::vector<std::vector<std::uint32_t>> entries,
           std::size_t m, std::size_t elem_size,
           std::size_t plaintext_bits);

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

 private:
  std::vector<std::vector<std::uint32_t>> entries_;
  std::size_t m_;
  std::size_t elem_size_;
  std::size_t plaintext_bits_;
};

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_DATABASE_H_
