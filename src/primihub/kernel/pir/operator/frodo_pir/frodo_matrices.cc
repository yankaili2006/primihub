/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

#include <sstream>

namespace primihub::pir::frodo {

std::vector<std::vector<std::uint32_t>> SwapMatrixFmt(
    const std::vector<std::vector<std::uint32_t>>& matrix) {
  if (matrix.empty()) {
    return {};
  }
  const std::size_t height = matrix.size();
  const std::size_t width = matrix[0].size();
  std::vector<std::vector<std::uint32_t>> swapped(width);
  for (auto& col : swapped) {
    col.reserve(height);
  }
  for (const auto& row : matrix) {
    // Upstream contract: every row has the same width as matrix[0].
    // We mirror that — no defensive resize beyond width.
    for (std::size_t i = 0; i < width; ++i) {
      swapped[i].push_back(row[i]);
    }
  }
  return swapped;
}

std::vector<std::uint32_t> GetMatrixSecondAt(
    const std::vector<std::vector<std::uint32_t>>& matrix,
    std::size_t secidx) {
  if (matrix.empty() || secidx >= matrix[0].size()) {
    return {};
  }
  std::vector<std::uint32_t> col;
  col.reserve(matrix.size());
  for (const auto& row : matrix) {
    col.push_back(row[secidx]);
  }
  return col;
}

retcode VecMultU32U32(const std::vector<std::uint32_t>& row,
                      const std::vector<std::uint32_t>& col,
                      std::uint32_t* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "VecMultU32U32: out must be non-null";
    return retcode::FAIL;
  }
  if (row.size() != col.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "VecMultU32U32: row_len: " << row.size()
          << ", col_len: " << col.size()
          << ". Upstream raises ErrorUnexpectedInputSize here.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Upstream: for i in 0..row.len() {
  //   acc = acc.wrapping_add(row[i].wrapping_mul(col[i]));
  // }
  // C++ unsigned arithmetic is already wrap-around (mod 2^32) by
  // the language spec — no explicit wrap macros needed.
  std::uint32_t acc = 0u;
  for (std::size_t i = 0; i < row.size(); ++i) {
    acc += row[i] * col[i];
  }
  *out = acc;
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::frodo
