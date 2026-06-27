/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/backend/backend.h"

#include <gtest/gtest.h>
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/backend/avx2_backend.h"
#include "src/primihub/kernel/pir/operator/backend/cpu_backend.h"

namespace primihub::pir {
namespace {

TEST(PirBackendTest, CpuAlwaysAvailable) {
  CpuBackend b;
  EXPECT_TRUE(b.Available());
  EXPECT_EQ(b.Type(), Backend::CPU);
  EXPECT_EQ(b.Name(), "cpu");
}

TEST(PirBackendTest, Avx2PerfHintsExceedCpu) {
  // AVX2 hint must report higher throughput than CPU regardless of host
  // availability (perf hint is the algorithm's expectation, not a probe).
  CpuBackend cpu;
  Avx2Backend avx;
  auto cpu_hints = cpu.GetPerfHints();
  auto avx_hints = avx.GetPerfHints();
  EXPECT_GT(avx_hints.expected_throughput_qps_at_1e7,
            cpu_hints.expected_throughput_qps_at_1e7);
  EXPECT_LT(avx_hints.expected_latency_us_per_query_at_1e7,
            cpu_hints.expected_latency_us_per_query_at_1e7);
  EXPECT_TRUE(avx_hints.has_simd);
  EXPECT_FALSE(cpu_hints.has_simd);
}

TEST(PirBackendTest, SelectAutoOnCpuOnlyAlgoReturnsCpu) {
  // Algorithm only supports CPU — selector must return CPU regardless of
  // host AVX2/CUDA presence.
  std::set<Backend> only_cpu = {Backend::CPU};
  auto b = SelectBackend(Backend::AUTO, only_cpu);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->Type(), Backend::CPU);
}

TEST(PirBackendTest, SelectAutoPrefersAvx2WhenSupportedAndAvailable) {
  std::set<Backend> cpu_and_avx = {Backend::CPU, Backend::AVX2};
  auto b = SelectBackend(Backend::AUTO, cpu_and_avx);
  ASSERT_NE(b, nullptr);
  // If host has AVX2, we get AVX2; otherwise CPU. Either is acceptable, but
  // the chosen backend MUST be in the supported set and Available().
  EXPECT_TRUE(cpu_and_avx.count(b->Type()));
  EXPECT_TRUE(b->Available());
}

TEST(PirBackendTest, SelectExplicitCpuReturnsCpu) {
  std::set<Backend> supported = {Backend::CPU, Backend::AVX2};
  auto b = SelectBackend(Backend::CPU, supported);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->Type(), Backend::CPU);
}

TEST(PirBackendTest, SelectPreferredUnsupportedFallsBack) {
  // Caller prefers CUDA but algorithm only supports CPU+AVX2.
  std::set<Backend> supported = {Backend::CPU, Backend::AVX2};
  auto b = SelectBackend(Backend::CUDA, supported);
  ASSERT_NE(b, nullptr);
  EXPECT_TRUE(supported.count(b->Type()));
  EXPECT_NE(b->Type(), Backend::CUDA);
}

TEST(PirBackendTest, SelectEmptySupportedReturnsNull) {
  std::set<Backend> empty;
  auto b = SelectBackend(Backend::AUTO, empty);
  EXPECT_EQ(b, nullptr);
}

}  // namespace
}  // namespace primihub::pir
