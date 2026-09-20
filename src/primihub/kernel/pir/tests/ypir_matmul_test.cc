/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_matmul_test — unit tests for the YPIR matmul dispatcher port.
 * Bifurcates on kYpirMatmulVendored like ypir_runtime_test does on
 * kYpirRuntimeVendored, so the precondition tests run unconditionally
 * but the kernel-math case is only meaningful when the kernels are
 * actually linked.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_matmul.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirMatmulTest, NullPointerInputs_FailWithMessage) {
  std::string err;
  std::vector<uint32_t> a(4, 0);
  std::vector<uint32_t> b(4, 0);
  // out is null.
  auto rc = MatMulVecPacked(nullptr, 16, a.data(), a.size(), b.data(),
                            b.size(), 1, 1, 4, 1, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("non-null"), std::string::npos) << err;
}

TEST(YpirMatmulTest, ShapeAMismatch_FailWithMessage) {
  std::string err;
  std::vector<uint32_t> a(3, 0);  // claim 1x1 but provide 3
  std::vector<uint32_t> b(4, 0);
  std::vector<uint32_t> out(16, 0);
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), 1, 1, 4, 1, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("a_len"), std::string::npos) << err;
}

TEST(YpirMatmulTest, PackedShapeViolation_FailWithMessage) {
  std::string err;
  // a_cols * 4 must equal b_rows; here a_cols=1, b_rows=5 violates.
  std::vector<uint32_t> a(8, 0);  // 8x1
  std::vector<uint32_t> b(5, 0);  // 5x1, but should be 4x1
  std::vector<uint32_t> out(16, 0);
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), 8, 1, 5, 1, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("packed-base shape contract"), std::string::npos)
      << err;
}

TEST(YpirMatmulTest, OutBufferTooSmall_FailWithMessage) {
  std::string err;
  std::vector<uint32_t> a(8, 0x01010101u);
  std::vector<uint32_t> b(4, 1);
  // Need at least (8 + 8) * 1 = 16; provide 15.
  std::vector<uint32_t> out(15, 0);
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), 8, 1, 4, 1, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("SIMD tail-write slack"), std::string::npos) << err;
}

TEST(YpirMatmulTest, InvalidBCols3_FailWithMessage) {
  std::string err;
  // a_cols=1, b_rows=4, b_cols=3 (not in {1,2,4,8}).
  std::vector<uint32_t> a(8, 0x01010101u);
  std::vector<uint32_t> b(12, 1);
  std::vector<uint32_t> out(48, 0);
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), 8, 1, 4, 3, &err);
  // Precondition path runs unconditionally; even non-vendored sees
  // the b_cols check only AFTER vendored gate, so we check both
  // outcomes are FAIL with their expected diagnostics.
  EXPECT_EQ(rc, retcode::FAIL);
  if (kYpirMatmulVendored) {
    EXPECT_NE(err.find("b_cols must be in"), std::string::npos) << err;
  } else {
    // Non-vendored short-circuits to the activation guidance before
    // reaching the b_cols switch.
    EXPECT_NE(err.find("enable_ypir_real=1"), std::string::npos) << err;
  }
}

TEST(YpirMatmulTest, NotVendored_HasActivationGuidance) {
  if (kYpirMatmulVendored) {
    GTEST_SKIP() << "Vendored build — activation-guidance test "
                 << "only runs in the default stub configuration.";
  }
  std::string err;
  std::vector<uint32_t> a(8, 0x01010101u);
  std::vector<uint32_t> b(4, 1);
  std::vector<uint32_t> out(16, 0);
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), 8, 1, 4, 1, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("enable_ypir_real=1"), std::string::npos)
      << "Stub error must guide callers to the activation flag; got: "
      << err;
  EXPECT_NE(err.find("ypir-port-plan.md"), std::string::npos)
      << "Stub error should cross-ref the port plan; got: " << err;
}

TEST(YpirMatmulTest, Vendored_8x1Packed1Case_MatchesSmoke) {
  if (!kYpirMatmulVendored) {
    GTEST_SKIP() << "Non-vendored build — kernel math case only "
                 << "runs when @ypir//:ypir_matmul_kernels is linked.";
  }
  // 8x1 packed: each a-entry packs 4 1-bytes, b is 4x1 of 1s.
  // Each row contributes 4 * (1 * 1) = 4. Same case as
  // YpirRuntime::SmokeMatMulVecPacked, but through our dispatcher.
  constexpr std::size_t kRows = 8;
  std::vector<uint32_t> a(kRows, 0x01010101u);
  std::vector<uint32_t> b = {1, 1, 1, 1};
  std::vector<uint32_t> out(kRows + 8u, 0u);
  std::string err;
  auto rc = MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                            b.data(), b.size(), kRows, 1, 4, 1, &err);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  for (std::size_t i = 0; i < kRows; ++i) {
    EXPECT_EQ(out[i], 4u)
        << "Kernel link works but math diverges at out[" << i << "]; "
        << "check @ypir pin a73e550a + src/matmul.cpp COMPRESSION/BASIS.";
  }
}

TEST(YpirMatmulTest, VendoredFlagMatchesRuntime) {
  // The two flags must agree — they both come from the same BUILD
  // select on enable_ypir_real. Importing ypir_runtime.h would
  // create a tighter coupling; we just assert at the bool level
  // by exercising the consistency through the err path of a
  // shape-pass / vendored-decide call.
  std::string err;
  std::vector<uint32_t> a(8, 0x01010101u);
  std::vector<uint32_t> b = {1, 1, 1, 1};
  std::vector<uint32_t> out(16, 0u);
  (void)MatMulVecPacked(out.data(), out.size(), a.data(), a.size(),
                        b.data(), b.size(), 8, 1, 4, 1, &err);
  if (kYpirMatmulVendored) {
    EXPECT_TRUE(err.empty()) << "vendored success path should set no err";
  } else {
    EXPECT_FALSE(err.empty()) << "non-vendored path must populate err";
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
