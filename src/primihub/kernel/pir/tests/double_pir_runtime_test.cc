/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * double_pir_runtime_test — end-to-end smoke for the DoublePirRuntime
 * facade. Behaviour bifurcates on PIR_DOUBLE_PIR_RUNTIME_VENDORED,
 * matching the spiral_runtime_test pattern. The operator BUILD's
 * select() wires the define in or out, so a single test file covers
 * both compile modes by inspecting kDoublePirRuntimeVendored at run
 * time.
 *
 *   * Vendored mode (--define=enable_double_pir_real=1): SmokeMatMulVec
 *     MUST return SUCCESS — it calls @simplepir's matMulVec on a 4x4
 *     known matrix and verifies the C output matches the in-line
 *     computed expected values.
 *
 *   * Stub mode (default): SmokeMatMulVec MUST return FAIL with a
 *     populated err string mentioning the activation flag, so
 *     accidentally-default builds still surface a clear actionable
 *     diagnostic.
 */
#include <gtest/gtest.h>

#include <string>

#include "src/primihub/kernel/pir/operator/double_pir/double_pir_runtime.h"

namespace primihub::pir::double_pir {
namespace {

TEST(DoublePirRuntimeTest, SmokeMatchesVendoredFlag) {
  std::string err;
  auto rc = DoublePirRuntime::Instance().SmokeMatMulVec(&err);

  if (kDoublePirRuntimeVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS)
        << "Vendored build but smoke failed: " << err;
    EXPECT_TRUE(err.empty())
        << "SUCCESS path should not write an err message; got: " << err;
  } else {
    EXPECT_EQ(rc, retcode::FAIL)
        << "Default build should be in stub mode and return FAIL.";
    EXPECT_NE(err.find("enable_double_pir_real=1"), std::string::npos)
        << "Stub error message must guide callers to the activation "
           "flag; got: "
        << err;
  }
}

TEST(DoublePirRuntimeTest, SmokeIsIdempotent) {
  // Singleton + pure C kernels — calling smoke twice in a row must not
  // change the outcome. Catches any accidental statefulness sneaking
  // into the runtime.
  std::string err1, err2;
  auto rc1 = DoublePirRuntime::Instance().SmokeMatMulVec(&err1);
  auto rc2 = DoublePirRuntime::Instance().SmokeMatMulVec(&err2);
  EXPECT_EQ(rc1, rc2);
  EXPECT_EQ(err1, err2);
}

}  // namespace
}  // namespace primihub::pir::double_pir
