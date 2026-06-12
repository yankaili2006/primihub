/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

#include <sstream>

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

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



std::vector<std::vector<std::uint32_t>> GenerateLweMatrixFromSeed(
    const SeedBytes& seed, std::size_t lwe_dim, std::size_t width) {
  // Upstream:
  //   let mut a = Vec::with_capacity(width);
  //   let mut rng = get_seeded_rng(seed);
  //   for _ in 0..width {
  //     let mut v = Vec::with_capacity(lwe_dim);
  //     for _ in 0..lwe_dim { v.push(rng.next_u32()); }
  //     a.push(v);
  //   }
  //   a
  SeededRng rng(seed);
  std::vector<std::vector<std::uint32_t>> a;
  a.reserve(width);
  for (std::size_t col = 0; col < width; ++col) {
    std::vector<std::uint32_t> v;
    v.reserve(lwe_dim);
    for (std::size_t row = 0; row < lwe_dim; ++row) {
      v.push_back(rng.NextU32());
    }
    a.push_back(std::move(v));
  }
  return a;
}



namespace {

// Upstream:
//   const TERNARY_INTERVAL_SIZE: u32 = (u32::MAX - 2) / 3;
//   const TERNARY_REJECTION_SAMPLING_MAX: u32 = TERNARY_INTERVAL_SIZE * 3;
// We mirror these byte-for-byte.
constexpr std::uint32_t kTernaryIntervalSize =
    (0xFFFFFFFFu - 2u) / 3u;
constexpr std::uint32_t kTernaryRejectionMax =
    kTernaryIntervalSize * 3u;

}  // namespace

std::uint32_t RandomTernary() {
  std::uint32_t val = os_rng::NextU32();
  while (val > kTernaryRejectionMax) {
    val = os_rng::NextU32();
  }
  // val now in [0, 3*kTernaryIntervalSize]; trichotomy mirrors
  // upstream (note inclusive lower-bound on the first interval).
  if (val > kTernaryIntervalSize &&
      val <= kTernaryIntervalSize * 2u) {
    return 1u;
  }
  if (val > kTernaryIntervalSize * 2u) {
    return 0xFFFFFFFFu;  // upstream uses u32::MAX (= -1 mod 2^32)
  }
  return 0u;
}

std::vector<std::uint32_t> RandomTernaryVector(std::size_t width) {
  std::vector<std::uint32_t> out;
  out.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    out.push_back(RandomTernary());
  }
  return out;
}

}  // namespace primihub::pir::frodo
