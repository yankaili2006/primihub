/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::core::GaussianSampler — discrete Gaussian sampler
 * for LWE noise generation in SimplePIR / DoublePIR. Ports upstream
 * simplepir's pir/gauss.go (rejection sampling over a CDF table for
 * sigma = 6.4, modelled on Martin Albrecht's dgs library).
 *
 * Output is a signed int64 in approximately [-128, 128]; the upstream
 * CDF table has 129 entries with the last value 1.4e-87, so the
 * effective support is well within int64 range. Sign is uniformly
 * random (50/50 ±x except x=0).
 *
 * The CDF table is pinned to upstream simplepir@e9020b03 and assumes
 * the LWE sigma in kLweParamEntries[] (6.4). If a future upstream pin
 * uses a different sigma, both the WORKSPACE_GITHUB pin AND this
 * sampler's CDF table need to be re-derived — bump both in the same
 * commit.
 *
 * RNG ownership — the sampler does NOT own its RNG. Callers inject a
 * std::mt19937_64 reference so tests can seed deterministically while
 * production callers use std::random_device for entropy. Real LWE
 * security needs a cryptographically-strong RNG (e.g., a CSPRNG seeded
 * from /dev/urandom); std::mt19937_64 is fine for unit tests and
 * functional correctness but MUST be replaced for any production
 * security claim — flagged again in the class comment.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_GAUSSIAN_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_GAUSSIAN_H_

#include <cstdint>
#include <random>

namespace primihub::pir::core {

// Number of entries in the embedded CDF table.
extern const std::size_t kGaussianCdfTableSize;
extern const double kGaussianCdfTable[];

// Sigma the embedded CDF table was derived for. Exported for tests
// and parameter consistency checks (e.g., assert that LweParams sigma
// matches this constant before using the sampler).
constexpr double kGaussianSigma = 6.4;

// Stateless rejection sampler. Holds a reference to a caller-owned
// std::mt19937_64. NOT cryptographically strong — std::mt19937 is a
// statistical RNG. Production code that depends on LWE security MUST
// inject a CSPRNG-backed source instead.
class GaussianSampler {
 public:
  explicit GaussianSampler(std::mt19937_64& rng) : rng_(rng) {}

  // Returns a single sample drawn from the discrete Gaussian of
  // stddev kGaussianSigma. Expected output range is approximately
  // [-128, 128] given the embedded table.
  int64_t Sample();

 private:
  std::mt19937_64& rng_;
};

}  // namespace primihub::pir::core

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_GAUSSIAN_H_
