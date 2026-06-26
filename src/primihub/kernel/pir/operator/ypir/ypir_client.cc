/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"

namespace primihub::pir::ypir {

std::vector<std::uint32_t> GenerateMatrixRing(ChaChaRng& rng_pub, std::size_t n,
                                              std::size_t rows,
                                              std::size_t cols) {
  assert(rows % n == 0);
  assert(cols % n == 0);
  const std::size_t rows_outer = rows / n;
  const std::size_t cols_outer = cols / n;

  std::vector<std::uint32_t> out(rows * cols, 0);
  for (std::size_t i = 0; i < rows_outer; ++i) {
    for (std::size_t j = 0; j < cols_outer; ++j) {
      std::vector<std::uint32_t> a(n);
      for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng_pub.NextU32();
      const std::vector<std::uint32_t> mat = NegacyclicMatrixU32(a);
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          out[(i * n + k) * cols + (j * n + l)] = mat[k * n + l];
    }
  }
  return out;
}

std::vector<PolyMatrixRaw> YClient::GenerateQueryImpl(
    std::uint8_t public_seed_idx, std::size_t dim_log2, bool packing,
    std::size_t index, ChaChaRng& noise_rng) const {
  const Params& p = ctx_->params();
  ChaChaRng rng_pub = ChaChaRng::FromSeed(GetSeed(public_seed_idx));
  const std::uint64_t scale_k = p.modulus / p.pt_modulus;
  const std::uint64_t factor = InvertUintMod(p.poly_len, p.modulus);

  std::vector<PolyMatrixRaw> out;
  const std::size_t count = static_cast<std::size_t>(1) << dim_log2;
  out.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    PolyMatrixRaw scalar = ctx_->ZeroRaw(1, 1);
    if (i == index / p.poly_len) scalar.data[index % p.poly_len] = scale_k;
    if (packing) {
      // scalar *= single_value(factor): a constant-poly multiply is just a
      // per-coefficient scaling by factor (mod modulus).
      for (std::size_t z = 0; z < p.poly_len; ++z)
        scalar.data[z] = MultiplyUintMod(scalar.data[z], factor, p.modulus);
    }
    // multiply_ct == true upstream: always the scaled encryption.
    const PolyMatrixNTT ct = client_->EncryptMatrixScaledReg(
        ctx_->ToNtt(scalar), noise_rng, rng_pub, factor);
    out.push_back(ctx_->FromNtt(ct));
  }
  return out;
}

std::vector<std::uint64_t> YClient::DecodeResponse(
    const std::vector<std::uint64_t>& response, std::size_t db_cols) const {
  const Params& p = ctx_->params();
  const std::uint64_t* sk = client_->SkReg().data.data();  // poly_len coeffs

  std::vector<std::uint64_t> out(db_cols);
  for (std::size_t col = 0; col < db_cols; ++col) {
    __uint128_t sum = 0;
    for (std::size_t i = 0; i < p.poly_len; ++i)
      sum += static_cast<__uint128_t>(response[i * db_cols + col]) * sk[i];
    sum += response[p.poly_len * db_cols + col];
    const std::uint64_t result = static_cast<std::uint64_t>(sum % p.modulus);
    out[col] = Rescale(result, p.modulus, p.pt_modulus);
  }
  return out;
}

}  // namespace primihub::pir::ypir
