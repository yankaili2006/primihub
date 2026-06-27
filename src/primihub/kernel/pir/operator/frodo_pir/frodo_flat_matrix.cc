/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_flat_matrix.h"

#include <cassert>

namespace primihub::pir::frodo {

ColMajorMatrix::ColMajorMatrix(std::size_t height, std::size_t width)
    : height_(height), width_(width), storage_(width * height, 0u) {}

ColMajorMatrix::ColMajorMatrix(std::size_t height, std::size_t width,
                               NoInit)
    : height_(height), width_(width) {
  // resize value-initialises uint32_t to 0 on libstdc++ today. The
  // NoInit tag documents that the caller is about to overwrite
  // every byte (e.g. FillBytesBulk in chunk g-2), and lets a
  // future revision plug in an uninitialised allocator without
  // changing the call site.
  storage_.resize(width * height);
}

std::uint32_t& ColMajorMatrix::at(std::size_t col, std::size_t row) {
  assert(col < width_ && "ColMajorMatrix::at: col out of range");
  assert(row < height_ && "ColMajorMatrix::at: row out of range");
  if (col >= width_ || row >= height_) {
    // Soft boundary on release builds — return a dangling ref to
    // storage_[0] rather than UB-accessing past the end. Matches
    // frodo_database::VecMult / GetRow soft-OOB convention so
    // callers ported from vector<vector<u32>> see the same
    // behaviour.
    static std::uint32_t sink = 0;
    sink = 0;
    return sink;
  }
  return storage_[col * height_ + row];
}

const std::uint32_t& ColMajorMatrix::at(std::size_t col,
                                        std::size_t row) const {
  assert(col < width_ && "ColMajorMatrix::at: col out of range");
  assert(row < height_ && "ColMajorMatrix::at: row out of range");
  if (col >= width_ || row >= height_) {
    static const std::uint32_t sink = 0;
    return sink;
  }
  return storage_[col * height_ + row];
}

std::uint32_t* ColMajorMatrix::column_data(std::size_t col) {
  assert(col < width_ && "ColMajorMatrix::column_data: col out of range");
  if (col >= width_) return nullptr;
  return storage_.data() + col * height_;
}

const std::uint32_t* ColMajorMatrix::column_data(std::size_t col) const {
  assert(col < width_ && "ColMajorMatrix::column_data: col out of range");
  if (col >= width_) return nullptr;
  return storage_.data() + col * height_;
}

bool ColMajorMatrix::operator==(const ColMajorMatrix& o) const {
  return height_ == o.height_ && width_ == o.width_ &&
         storage_ == o.storage_;
}

}  // namespace primihub::pir::frodo
