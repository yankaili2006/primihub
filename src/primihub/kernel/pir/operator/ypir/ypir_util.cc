/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

namespace primihub::pir::ypir {

std::vector<std::uint64_t> NegacyclicPermU64Mod(
    const std::vector<std::uint64_t>& a, std::size_t shift,
    std::uint64_t modulus) {
  const std::size_t n = a.size();
  if (n == 0) {
    return {};
  }
  // Upstream Rust does not bounds-check `a[shift - i]` — passing
  // `shift >= n` would panic via the slice indexing. We treat that
  // as a caller error and return an empty vector at the boundary.
  if (shift >= n) {
    return {};
  }
  std::vector<std::uint64_t> out(n, 0u);
  for (std::size_t i = 0; i <= shift; ++i) {
    out[i] = a[shift - i];
  }
  for (std::size_t i = shift + 1; i < n; ++i) {
    // Upstream:
    //   out[i] = modulus - (a[n - (i - shift)] % modulus);
    //   if out[i] == modulus { out[i] = 0; }
    //
    // The `% modulus` matters when a[*] >= modulus (otherwise
    // identical to modulus - a[*]); the `if == modulus` clamp
    // converts `modulus - 0 == modulus` back to 0.
    std::uint64_t a_val = a[n - (i - shift)] % modulus;
    std::uint64_t v = modulus - a_val;
    if (v == modulus) {
      v = 0;
    }
    out[i] = v;
  }
  return out;
}

std::vector<std::uint64_t> NegacyclicMatrixU64Mod(
    const std::vector<std::uint64_t>& a, std::uint64_t modulus) {
  const std::size_t n = a.size();
  if (n == 0) {
    return {};
  }
  std::vector<std::uint64_t> out(n * n, 0u);
  for (std::size_t i = 0; i < n; ++i) {
    auto perm = NegacyclicPermU64Mod(a, i, modulus);
    // perm size is n when shift < n (guaranteed here).
    for (std::size_t j = 0; j < n; ++j) {
      out[j * n + i] = perm[j];
    }
  }
  return out;
}

std::vector<std::uint64_t> ConcatHorizontalU64(
    const std::vector<std::vector<std::uint64_t>>& v_a,
    std::size_t a_rows, std::size_t a_cols) {
  if (v_a.empty() || a_rows == 0 || a_cols == 0) {
    return {};
  }
  // Validate every input matrix has exactly a_rows * a_cols entries.
  // Upstream Rust panics on out-of-range slice access; we prefer a
  // soft empty return.
  for (const auto& m : v_a) {
    if (m.size() != a_rows * a_cols) {
      return {};
    }
  }
  std::vector<std::uint64_t> out(a_rows * a_cols * v_a.size(), 0u);
  for (std::size_t i = 0; i < a_rows; ++i) {
    for (std::size_t j = 0; j < a_cols; ++j) {
      for (std::size_t k = 0; k < v_a.size(); ++k) {
        const std::size_t idx = i * a_cols + j;
        const std::size_t out_idx =
            i * a_cols * v_a.size() + k * a_cols + j;
        out[out_idx] = v_a[k][idx];
      }
    }
  }
  return out;
}

}  // namespace primihub::pir::ypir
