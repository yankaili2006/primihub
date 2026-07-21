/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_discrete_gaussian — port of upstream spiral-rs
 * src/discrete_gaussian.rs (the sampler YPIR's lwe.rs encrypt path
 * depends on). Chunk 4b of the YPIR port (see
 * docs/pir/ypir-port-plan.md); unblocks the lwe.rs encrypt port.
 *
 * Scope of this chunk (Spiral-free / RNG-agnostic subset):
 *   * Init(noise_width)            — build the inverse-CDF table
 *   * Sample(modulus, sampled_val) — constant-time inverse-CDF lookup
 * Deferred:
 *   * fast_sample (WeightedIndex, non-constant-time, unused by encrypt)
 *   * sample_matrix (operates on Spiral PolyMatrixRaw — chunk 11)
 *
 * RNG decoupling: upstream `sample(modulus, rng)` does
 * `let sampled_val = rng.gen::<u64>(); <inverse-CDF lookup>`. We take
 * `sampled_val` as a parameter so the (security-critical) lookup is
 * pure and deterministically testable; the caller draws the uniform
 * u64 from the ChaCha20 stream when lwe encrypt is assembled.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_DISCRETE_GAUSSIAN_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_DISCRETE_GAUSSIAN_H_

#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// NUM_WIDTHS from upstream: the table spans +/- ceil(noise_width * 4).
inline constexpr int kNumWidths = 4;

class DiscreteGaussian {
 public:
  // Mirror of spiral-rs DiscreteGaussian::init: max_val =
  // ceil(noise_width * kNumWidths); for each integer i in
  // [-max_val, max_val] the unnormalised weight is
  // exp(-PI * i^2 / noise_width^2); cdf_table[k] is the cumulative
  // normalised probability through the k-th integer, scaled by 2^64
  // and rounded (saturating at UINT64_MAX, matching Rust `as u64`).
  static DiscreteGaussian Init(double noise_width);

  // Constant-time inverse-CDF sample. Returns the noise value reduced
  // mod `modulus` (negatives wrap to modulus + v), exactly as
  // spiral-rs DiscreteGaussian::sample given `sampled_val =
  // rng.gen::<u64>()`. The result equals (i_min - max_val) mod modulus
  // where i_min is the smallest index with cdf_table[i] >= sampled_val.
  std::uint64_t Sample(std::uint64_t modulus,
                       std::uint64_t sampled_val) const;

  std::int64_t max_val() const { return max_val_; }
  const std::vector<std::uint64_t>& cdf_table() const { return cdf_table_; }

 private:
  std::vector<std::uint64_t> cdf_table_;
  std::int64_t max_val_ = 0;
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_DISCRETE_GAUSSIAN_H_
