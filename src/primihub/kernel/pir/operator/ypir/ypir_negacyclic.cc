/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

namespace primihub::pir::ypir {

std::vector<std::uint32_t> NegacyclicMatrixU32(
    const std::vector<std::uint32_t>& b) {
  const std::size_t n = b.size();
  std::vector<std::uint32_t> res(n * n, 0u);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      // b[(n + i - j) % n] — Rust's `(n + i - j) % n` keeps the
      // arithmetic in u32; C++ size_t is unsigned and has the same
      // wrap-around behaviour for `(n + i - j)` as long as we don't
      // promote.
      std::uint32_t b_val = b[(n + i - j) % n];
      if (i < j) {
        // wrapping_neg on u32 — C++ unsigned negation is well-defined
        // modular arithmetic.
        b_val = static_cast<std::uint32_t>(0u - b_val);
      }
      // upstream comment: `nb: transposed` — column-major.
      res[j * n + i] = b_val;
    }
  }
  return res;
}

std::vector<std::uint32_t> NaiveNegacyclicConvolveU32(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b) {
  if (a.size() != b.size()) {
    return {};
  }
  const std::size_t n = a.size();
  std::vector<std::uint32_t> res(n, 0u);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      std::uint32_t b_val = b[(n + i - j) % n];
      if (i < j) {
        b_val = static_cast<std::uint32_t>(0u - b_val);
      }
      // Upstream uses `+=` on u32; in Rust release-mode this wraps,
      // and that's the intended modular semantics. C++ unsigned `+`
      // also wraps; `+=` here matches upstream byte-for-byte.
      res[i] = res[i] + a[j] * b_val;
    }
  }
  return res;
}

std::vector<std::uint32_t> NegacyclicPermU32(
    const std::vector<std::uint32_t>& a) {
  const std::size_t n = a.size();
  if (n == 0) {
    return {};
  }
  std::vector<std::uint32_t> res(n, 0u);
  res[0] = a[0];
  for (std::size_t i = 1; i < n; ++i) {
    res[i] = static_cast<std::uint32_t>(0u - a[(n - i) % n]);
  }
  return res;
}

}  // namespace primihub::pir::ypir
