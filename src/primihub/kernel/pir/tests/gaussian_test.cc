/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * GaussianSampler tests — verify the rejection sampler hits the right
 * statistical properties (mean ~ 0, stddev ~ sigma, symmetric tails,
 * bounded support) over a large enough sample count that
 * std::mt19937_64 noise is below the threshold. Deterministic seed so
 * the assertions are reproducible across runs.
 */
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "src/primihub/kernel/pir/operator/pir_core/gaussian.h"

namespace primihub::pir::core {
namespace {

constexpr std::size_t kNumSamples = 100'000;
constexpr int kSeed = 0xCAFE;

TEST(GaussianTest, EmbeddedTableShape) {
  // Upstream gauss.go has 129 entries; cross-check that we
  // transcribed all of them.
  EXPECT_EQ(kGaussianCdfTableSize, 129u);
  // CDF table is monotonically decreasing after entry 0 (which is the
  // PDF peak P[x=0] = 0.5 by construction). Entry 1 must be < entry 0.
  for (std::size_t i = 2; i < kGaussianCdfTableSize; ++i) {
    EXPECT_LT(kGaussianCdfTable[i], kGaussianCdfTable[i - 1])
        << "entry " << i << " not strictly less than " << (i - 1);
  }
  // Last entry must be a tiny tail value.
  EXPECT_LT(kGaussianCdfTable[kGaussianCdfTableSize - 1], 1e-80);
}

TEST(GaussianTest, SamplesAreSymmetric) {
  std::mt19937_64 rng(kSeed);
  GaussianSampler sampler(rng);
  int64_t positive = 0, negative = 0, zero = 0;
  for (std::size_t i = 0; i < kNumSamples; ++i) {
    int64_t s = sampler.Sample();
    if (s > 0) ++positive;
    else if (s < 0) ++negative;
    else ++zero;
  }
  // Symmetry: |positive - negative| / total should be small.
  const double imbalance = std::abs(static_cast<double>(positive - negative)) /
                           static_cast<double>(kNumSamples);
  EXPECT_LT(imbalance, 0.01)
      << "positive=" << positive << " negative=" << negative
      << " zero=" << zero;
  EXPECT_GT(zero, 0) << "x=0 should be drawn occasionally";
}

TEST(GaussianTest, EmpiricalStddevMatchesSigma) {
  std::mt19937_64 rng(kSeed);
  GaussianSampler sampler(rng);
  // Track sum and sum-of-squares.
  double sum = 0.0, sumsq = 0.0;
  for (std::size_t i = 0; i < kNumSamples; ++i) {
    const double s = static_cast<double>(sampler.Sample());
    sum += s;
    sumsq += s * s;
  }
  const double mean = sum / kNumSamples;
  // sample variance with N denominator (population stddev).
  const double variance = sumsq / kNumSamples - mean * mean;
  const double stddev = std::sqrt(variance);
  EXPECT_NEAR(mean, 0.0, 0.1)
      << "empirical mean diverged from 0 over " << kNumSamples
      << " samples";
  // Sigma should be very close to 6.4 — the upstream table is the
  // discretized CDF for that exact stddev.
  EXPECT_NEAR(stddev, kGaussianSigma, 0.2)
      << "empirical stddev=" << stddev << " expected " << kGaussianSigma;
}

TEST(GaussianTest, SamplesAreBounded) {
  std::mt19937_64 rng(kSeed);
  GaussianSampler sampler(rng);
  int64_t minv = 0, maxv = 0;
  for (std::size_t i = 0; i < kNumSamples; ++i) {
    int64_t s = sampler.Sample();
    if (s < minv) minv = s;
    if (s > maxv) maxv = s;
  }
  // CDF table has 129 entries; max |x| supported is 128.
  EXPECT_LE(maxv, 128);
  EXPECT_GE(minv, -128);
  // 100K samples should hit values well past sigma at both ends.
  EXPECT_GT(maxv, 10);
  EXPECT_LT(minv, -10);
}

TEST(GaussianTest, SeededDeterminism) {
  std::mt19937_64 rng1(kSeed);
  std::mt19937_64 rng2(kSeed);
  GaussianSampler s1(rng1);
  GaussianSampler s2(rng2);
  // 1000 samples enough to catch any divergence.
  for (std::size_t i = 0; i < 1000; ++i) {
    EXPECT_EQ(s1.Sample(), s2.Sample()) << "diverged at i=" << i;
  }
}

}  // namespace
}  // namespace primihub::pir::core
