/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe.h"

#include <cstddef>

#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"

namespace primihub::pir::ypir {

std::vector<std::uint32_t> LweEncrypt(const std::vector<std::uint32_t>& sk,
                                      const std::vector<std::uint32_t>& a,
                                      std::uint32_t pt, std::uint32_t e) {
  const std::size_t n = sk.size();
  std::vector<std::uint32_t> ct;
  ct.reserve(n + 1);
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ct.push_back(a[i]);
    sum += a[i] * sk[i];  // u32 wrapping
  }
  // b = wrapping_neg(sum) + pt + e
  const std::uint32_t b = (0u - sum) + pt + e;
  ct.push_back(b);
  return ct;
}

std::vector<std::uint32_t> LweEncryptMany(
    const std::vector<std::uint32_t>& sk,
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& v_pt,
    const std::vector<std::uint32_t>& e) {
  const std::size_t n = sk.size();
  std::vector<std::uint32_t> nega_a = NegacyclicMatrixU32(a);  // n*n row-major
  std::vector<std::uint32_t> last_row(n, 0);
  for (std::size_t col = 0; col < n; ++col) {
    std::uint32_t sum = 0;
    for (std::size_t row = 0; row < n; ++row) {
      sum += nega_a[row * n + col] * sk[row];  // u32 wrapping
    }
    last_row[col] = (0u - sum) + v_pt[col] + e[col];
  }
  std::vector<std::uint32_t> ct = std::move(nega_a);
  ct.insert(ct.end(), last_row.begin(), last_row.end());
  return ct;
}

std::uint32_t LweDecrypt(const std::vector<std::uint32_t>& sk,
                         const std::vector<std::uint32_t>& ct) {
  const std::size_t n = sk.size();
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sum += ct[i] * sk[i];  // u32 wrapping
  }
  sum += ct[n];
  return sum;
}

}  // namespace primihub::pir::ypir
