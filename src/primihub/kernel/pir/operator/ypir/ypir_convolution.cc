/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution.h"

namespace primihub::pir::ypir {

std::vector<std::uint32_t> NaiveNegacyclicConvolve(
    const std::vector<std::uint32_t>& a, const std::vector<std::uint32_t>& b) {
  const std::size_t n = a.size();
  std::vector<std::uint32_t> res(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::uint32_t b_val = b[(n + i - j) % n];
      if (i < j) b_val = 0u - b_val;  // wrapping_neg
      res[i] += a[j] * b_val;         // u32 wrapping
    }
  }
  return res;
}

std::vector<std::uint32_t> NaiveMultiplyMatrices(
    const std::vector<std::uint32_t>& a, std::size_t a_rows, std::size_t a_cols,
    const std::vector<std::uint32_t>& b, std::size_t b_rows, std::size_t b_cols,
    bool is_b_transposed) {
  // a_cols must equal b_rows (contracted dimension).
  std::vector<std::uint32_t> result(a_rows * b_cols, 0);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t j = 0; j < b_cols; ++j) {
      for (std::size_t k = 0; k < a_cols; ++k) {
        const std::size_t a_idx = i * a_cols + k;
        const std::size_t b_idx =
            is_b_transposed ? (j * b_rows + k) : (k * b_cols + j);
        result[i * b_cols + j] += a[a_idx] * b[b_idx];  // u32 wrapping
      }
    }
  }
  return result;
}

}  // namespace primihub::pir::ypir
