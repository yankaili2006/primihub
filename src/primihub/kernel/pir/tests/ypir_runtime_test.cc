/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_runtime_test — end-to-end smoke for the YpirRuntime facade.
 * Behaviour bifurcates on kYpirRuntimeVendored. Same shape as
 * double_pir_runtime_test (commit dc037df7).
 */
#include <gtest/gtest.h>

#include <string>

#include "src/primihub/kernel/pir/operator/ypir/ypir_runtime.h"

namespace primihub::pir::ypir {
namespace {

TEST(YpirRuntimeTest, SmokeMatchesVendoredFlag) {
  std::string err;
  auto rc = YpirRuntime::Instance().SmokeMatMulVecPacked(&err);
  if (kYpirRuntimeVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS)
        << "Vendored build but smoke failed: " << err;
    EXPECT_TRUE(err.empty());
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_ypir_real=1"), std::string::npos)
        << "Stub error must guide callers to the activation flag; got: "
        << err;
  }
}

TEST(YpirRuntimeTest, SmokeIsIdempotent) {
  std::string err1, err2;
  auto rc1 = YpirRuntime::Instance().SmokeMatMulVecPacked(&err1);
  auto rc2 = YpirRuntime::Instance().SmokeMatMulVecPacked(&err2);
  EXPECT_EQ(rc1, rc2);
  EXPECT_EQ(err1, err2);
}

}  // namespace
}  // namespace primihub::pir::ypir
