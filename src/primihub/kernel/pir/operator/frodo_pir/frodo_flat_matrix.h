/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_flat_matrix — column-major dense matrix of uint32_t backed
 * by ONE flat std::vector<uint32_t>. Replaces the
 * std::vector<std::vector<uint32_t>> form used by the FrodoPIR
 * Setup-time matrices (LWE matrix A, BaseParams::rhs_,
 * CommonParams::matrix_, Database::entries_) so the dominant
 * SwapMatrixFmt cost (5.7 s @ N=1M in chunk-1230237b profiling)
 * stops being bottlenecked on per-column allocation + the 2 GB
 * pre-init pass that `vector<vector<u32>>(width, vector<u32>(height))`
 * triggers on libstdc++.
 *
 * Position in the FrodoPIR port plan (task 7.1):
 *   chunk g-1 — THIS chunk. Pure additive: defines the type +
 *               tests. No existing function is modified.
 *   chunk g-2 — GenerateLweMatrixFromSeedFlat overload that
 *               returns ColMajorMatrix.
 *   chunk g-3 — SwapMatrixFmtFlat overload (2D-tiled transpose
 *               on flat memory).
 *   chunk g-4 — migrate Database::entries_.
 *   chunk g-5 — migrate BaseParams::rhs_ + CommonParams::matrix_.
 *   chunk g-6 — drop the vector<vector<u32>> overloads.
 *   chunk g-7 — bench + OpenSpec sync.
 *
 * Memory layout: column-major. Element at (col, row) lives at
 *   storage_[col * height_ + row]
 * which means column c occupies a contiguous run of `height_`
 * u32s — directly usable as a VecMultU32U32 operand without copy.
 *
 * Convention vs. prior std::vector<std::vector<uint32_t>>:
 *   - mat.at(c, r)         <-> old `mat[c][r]`
 *   - mat.column_data(c)   <-> old `mat[c].data()`
 *   - mat.width()          <-> old `mat.size()`
 *   - mat.height()         <-> old `mat[0].size()` (under the
 *                              upstream-mirrored uniform-width
 *                              contract).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FLAT_MATRIX_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FLAT_MATRIX_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::frodo {

class ColMajorMatrix {
 public:
  // Tag for the uninitialised-storage constructor. The caller
  // PROMISES to overwrite every byte before reading. Today the
  // implementation matches the regular ctor (std::vector::resize
  // value-initialises arithmetic types), but the tag exists so a
  // future revision can switch to an allocator that skips the
  // zero-fill without an API break — and so reviewers reading
  // call sites can see that the caller knows initial contents
  // are garbage.
  struct NoInit {};

  ColMajorMatrix() = default;

  // Zero-initialised height x width matrix. One contiguous
  // allocation of `width * height` u32s (no per-column alloc).
  ColMajorMatrix(std::size_t height, std::size_t width);

  // Same shape but the storage is value-initialised today, with
  // the contract that callers MUST overwrite it. See NoInit.
  ColMajorMatrix(std::size_t height, std::size_t width, NoInit);

  std::size_t height() const { return height_; }
  std::size_t width() const { return width_; }
  bool empty() const { return storage_.empty(); }

  // Element access. Preconditions:
  //   col < width(), row < height().
  // On debug builds the assert fires; on release we return a
  // dangling reference to storage_[0] for OOB (matches the
  // existing frodo_database VecMult / GetRow soft-boundary
  // convention rather than throwing).
  std::uint32_t& at(std::size_t col, std::size_t row);
  const std::uint32_t& at(std::size_t col, std::size_t row) const;

  // Pointer to the first u32 of column `col`. Column c spans
  // [column_data(c), column_data(c) + height()).
  std::uint32_t* column_data(std::size_t col);
  const std::uint32_t* column_data(std::size_t col) const;

  // Flat raw storage. Exposed so chunk g-2's
  // GenerateLweMatrixFromSeedFlat can call FillBytesBulk once
  // over the full `total_u32s() * sizeof(uint32_t)` byte range
  // instead of per-column.
  std::uint32_t* raw_data() { return storage_.data(); }
  const std::uint32_t* raw_data() const { return storage_.data(); }
  std::size_t total_u32s() const { return storage_.size(); }

  // Equality compares shape + element contents. Used by tests
  // that pin byte-for-byte equivalence between the new flat
  // form and a reference computed via the legacy
  // vector<vector<u32>> path.
  bool operator==(const ColMajorMatrix& o) const;
  bool operator!=(const ColMajorMatrix& o) const { return !(*this == o); }

 private:
  std::size_t height_ = 0;
  std::size_t width_ = 0;
  std::vector<std::uint32_t> storage_;  // size == width_ * height_
};

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FLAT_MATRIX_H_
