// Copyright 2026. test-only helpers bridging primihub::pir::frodo
// ColMajorMatrix (chunk g-1) and the std::vector<std::vector<uint32_t>>
// fixture form that pre-g-6 tests used. Provides explicit naming for
// the two pre-existing nested conventions found in this codebase:
//   * RowMajor: nested[r][c] = element at row r col c.  Used as the
//     INPUT to the dropped SwapMatrixFmt nested overload.
//   * ColMajor: nested[c][r] = element at row r col c.  Used as the
//     OUTPUT of dropped SwapMatrixFmt / GenerateLweMatrixFromSeed
//     nested overloads, and as Database::EntriesForTest pre-g-6.
// Header-only, test-translation-unit-local via inline linkage.
#ifndef SRC_PRIMIHUB_KERNEL_PIR_TESTS_FRODO_TEST_HELPERS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_TESTS_FRODO_TEST_HELPERS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_flat_matrix.h"

namespace primihub::pir::frodo::testing {

// rows[r][c] -> m.at(c, r); height = rows.size(), width = rows[0].size().
inline ColMajorMatrix MatrixFromRows(
    const std::vector<std::vector<std::uint32_t>>& rows) {
  if (rows.empty()) return {};
  const std::size_t h = rows.size();
  const std::size_t w = rows[0].size();
  ColMajorMatrix m(h, w);
  for (std::size_t r = 0; r < h; ++r) {
    for (std::size_t c = 0; c < w; ++c) {
      m.at(c, r) = rows[r][c];
    }
  }
  return m;
}

// cols[c][r] -> m.at(c, r); height = cols[0].size(), width = cols.size().
inline ColMajorMatrix MatrixFromCols(
    const std::vector<std::vector<std::uint32_t>>& cols) {
  if (cols.empty()) return {};
  const std::size_t w = cols.size();
  const std::size_t h = cols[0].size();
  ColMajorMatrix m(h, w);
  for (std::size_t c = 0; c < w; ++c) {
    for (std::size_t r = 0; r < h; ++r) {
      m.at(c, r) = cols[c][r];
    }
  }
  return m;
}

// Inverse of MatrixFromRows: rows[r][c] = m.at(c, r).
inline std::vector<std::vector<std::uint32_t>> RowsOfMatrix(
    const ColMajorMatrix& m) {
  std::vector<std::vector<std::uint32_t>> rows(
      m.height(), std::vector<std::uint32_t>(m.width()));
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      rows[r][c] = m.at(c, r);
    }
  }
  return rows;
}

// Inverse of MatrixFromCols: cols[c][r] = m.at(c, r).
inline std::vector<std::vector<std::uint32_t>> ColsOfMatrix(
    const ColMajorMatrix& m) {
  std::vector<std::vector<std::uint32_t>> cols(
      m.width(), std::vector<std::uint32_t>(m.height()));
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      cols[c][r] = m.at(c, r);
    }
  }
  return cols;
}

}  // namespace primihub::pir::frodo::testing

#endif  // SRC_PRIMIHUB_KERNEL_PIR_TESTS_FRODO_TEST_HELPERS_H_
