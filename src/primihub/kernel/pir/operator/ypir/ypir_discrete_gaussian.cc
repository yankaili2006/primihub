/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"

#include <cmath>
#include <cstddef>
#include <limits>

namespace primihub::pir::ypir {

namespace {

constexpr double kPi = 3.14159265358979311600;  // M_PI (IEEE-754 double)

// Constant-time mask: returns ~0ull when a <= b, else 0. Branch-free —
// (b - a) over 128 bits borrows (sets the high word to all-ones) iff
// a > b, mirroring subtle's ConstantTimeGreater negated.
inline std::uint64_t CtLeMask(std::uint64_t a, std::uint64_t b) {
  unsigned __int128 d =
      static_cast<unsigned __int128>(b) - static_cast<unsigned __int128>(a);
  std::uint64_t borrow = static_cast<std::uint64_t>(d >> 64);  // 0 or ~0
  return ~borrow;
}

}  // namespace

DiscreteGaussian DiscreteGaussian::Init(double noise_width) {
  DiscreteGaussian dg;
  dg.max_val_ = static_cast<std::int64_t>(std::ceil(noise_width * kNumWidths));

  // Unnormalised Gaussian weights over [-max_val, max_val].
  std::vector<double> table;
  table.reserve(static_cast<std::size_t>(2 * dg.max_val_ + 1));
  double total = 0.0;
  for (std::int64_t i = -dg.max_val_; i <= dg.max_val_; ++i) {
    const double di = static_cast<double>(i);
    const double p = std::exp(-kPi * di * di / (noise_width * noise_width));
    table.push_back(p);
    total += p;
  }

  // Cumulative normalised CDF, scaled to u64 with saturating round
  // (Rust `(cum * u64::MAX as f64).round() as u64` saturates floats
  //  >= 2^64 to u64::MAX; the C++ cast would be UB, so clamp).
  const double two_pow_64 = std::ldexp(1.0, 64);  // 2^64
  dg.cdf_table_.reserve(table.size());
  double cum_prob = 0.0;
  for (double p : table) {
    cum_prob += p / total;
    const double scaled = std::round(
        cum_prob * static_cast<double>(std::numeric_limits<std::uint64_t>::max()));
    const std::uint64_t v = (scaled >= two_pow_64)
        ? std::numeric_limits<std::uint64_t>::max()
        : static_cast<std::uint64_t>(scaled);
    dg.cdf_table_.push_back(v);
  }
  return dg;
}

std::uint64_t DiscreteGaussian::Sample(std::uint64_t modulus,
                                       std::uint64_t sampled_val) const {
  const std::size_t len = static_cast<std::size_t>(2 * max_val_ + 1);
  std::uint64_t to_output = 0;
  // Iterate indices downward; cdf_table is non-decreasing, so the last
  // write that sticks is the smallest i with sampled_val <= cdf_table[i].
  for (std::size_t step = 0; step < len; ++step) {
    const std::size_t i = len - 1 - step;
    std::int64_t out_val = static_cast<std::int64_t>(i) - max_val_;
    // Loop-index dependent, not secret-dependent (upstream notes this).
    if (out_val < 0) out_val += static_cast<std::int64_t>(modulus);
    const std::uint64_t out_val_u64 = static_cast<std::uint64_t>(out_val);

    const std::uint64_t mask = CtLeMask(sampled_val, cdf_table_[i]);
    to_output = (to_output & ~mask) | (out_val_u64 & mask);
  }
  return to_output;
}

}  // namespace primihub::pir::ypir
