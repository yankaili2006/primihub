/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

namespace primihub::pir::ypir {

namespace {

// floor(log2(x)) for x >= 1.
std::size_t Log2Floor(std::uint64_t x) {
  return 63 - static_cast<std::size_t>(__builtin_clzll(x));
}

// ceil(log2(x)) for x >= 1 (0 for x == 1).
std::uint64_t Log2Ceil(std::uint64_t x) {
  if (x <= 1) return 0;
  return 64 - static_cast<std::uint64_t>(__builtin_clzll(x - 1));
}

// a^{-1} mod m via the extended Euclidean algorithm.
std::uint64_t ModInverse(std::uint64_t a, std::uint64_t m) {
  std::int64_t t = 0, new_t = 1;
  std::int64_t r = static_cast<std::int64_t>(m), new_r = static_cast<std::int64_t>(a);
  while (new_r != 0) {
    const std::int64_t q = r / new_r;
    std::int64_t tmp = t - q * new_t;
    t = new_t;
    new_t = tmp;
    tmp = r - q * new_r;
    r = new_r;
    new_r = tmp;
  }
  if (t < 0) t += static_cast<std::int64_t>(m);
  return static_cast<std::uint64_t>(t);
}

}  // namespace

Params Params::Init(std::size_t poly_len, const std::vector<std::uint64_t>& moduli,
                    double noise_width, std::size_t n, std::uint64_t pt_modulus,
                    std::uint64_t q2_bits, std::size_t t_conv,
                    std::size_t t_exp_left, std::size_t t_exp_right,
                    std::size_t t_gsw, bool expand_queries, std::size_t db_dim_1,
                    std::size_t db_dim_2, std::size_t instances,
                    std::size_t db_item_size, std::size_t version) {
  Params p;
  p.poly_len = poly_len;
  p.poly_len_log2 = Log2Floor(static_cast<std::uint64_t>(poly_len));
  p.crt_count = moduli.size();

  p.modulus = 1;
  for (std::size_t i = 0; i < p.crt_count && i < kMaxModuli; ++i) {
    p.moduli[i] = moduli[i];
    p.modulus *= moduli[i];
    const auto crs = GetBarrettCrs(moduli[i]);
    p.barrett_cr_0[i] = crs.first;
    p.barrett_cr_1[i] = crs.second;
  }
  p.modulus_log2 = Log2Ceil(p.modulus);
  const auto crs_mod = GetBarrettCrs(p.modulus);
  p.barrett_cr_0_modulus = crs_mod.first;
  p.barrett_cr_1_modulus = crs_mod.second;

  if (p.crt_count == 2) {
    p.mod0_inv_mod1 = moduli[0] * ModInverse(moduli[0], moduli[1]);
    p.mod1_inv_mod0 = moduli[1] * ModInverse(moduli[1], moduli[0]);
  }

  p.noise_width = noise_width;
  p.n = n;
  p.pt_modulus = pt_modulus;
  p.q2_bits = q2_bits;
  p.t_conv = t_conv;
  p.t_exp_left = t_exp_left;
  p.t_exp_right = t_exp_right;
  p.t_gsw = t_gsw;
  p.expand_queries = expand_queries;
  p.db_dim_1 = db_dim_1;
  p.db_dim_2 = db_dim_2;
  p.instances = instances;
  p.db_item_size = db_item_size;
  p.version = version;
  return p;
}

}  // namespace primihub::pir::ypir
