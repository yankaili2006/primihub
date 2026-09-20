/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_transpose.h"

#include <cstring>

namespace primihub::pir::ypir {

std::vector<uint8_t> Transpose(const uint8_t* buf, std::size_t buf_len,
                               std::size_t rows, std::size_t cols,
                               std::size_t bytes_per_pt_el) {
  std::vector<uint8_t> out(buf_len, 0u);
  // Defensive: if the caller-provided buf_len is smaller than the
  // logical matrix size, bail with the zero-filled output. Upstream
  // Rust would panic via slice indexing; C++ aborts are worse than
  // returning empty in a wrapper.
  if (buf == nullptr || rows * cols * bytes_per_pt_el > buf_len) {
    return out;
  }
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const std::size_t src_off = (i * cols + j) * bytes_per_pt_el;
      const std::size_t dst_off = (j * rows + i) * bytes_per_pt_el;
      std::memcpy(out.data() + dst_off, buf + src_off, bytes_per_pt_el);
    }
  }
  return out;
}

bool TransposeF64(double* out, const double* a, std::size_t a_rows,
                  std::size_t a_cols) {
  static constexpr std::size_t kTile = 32;
  if (out == nullptr || a == nullptr) return false;
  if (a_rows < kTile || a_cols < kTile) return false;
  if (a_rows % kTile != 0 || a_cols % kTile != 0) return false;
  for (std::size_t i_outer = 0; i_outer < a_rows; i_outer += kTile) {
    for (std::size_t j_outer = 0; j_outer < a_cols; j_outer += kTile) {
      for (std::size_t i_inner = 0; i_inner < kTile; ++i_inner) {
        for (std::size_t j_inner = 0; j_inner < kTile; ++j_inner) {
          const std::size_t i = i_outer + i_inner;
          const std::size_t j = j_outer + j_inner;
          out[j * a_rows + i] = a[i * a_cols + j];
        }
      }
    }
  }
  return true;
}

}  // namespace primihub::pir::ypir
