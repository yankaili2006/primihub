/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * simple_pir_runtime_test — end-to-end smoke for the SimplePirRuntime
 * facade. Behaviour bifurcates on kSimplePirRuntimeVendored, set by
 * the operator BUILD's select(). Same shape as
 * double_pir_runtime_test (commit dc037df7).
 */
#include <gtest/gtest.h>

#include <string>

#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_runtime.h"

namespace primihub::pir::simple_pir {
namespace {

TEST(SimplePirRuntimeTest, SmokeMatchesVendoredFlag) {
  std::string err;
  auto rc = SimplePirRuntime::Instance().SmokeMatMul(&err);
  if (kSimplePirRuntimeVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS)
        << "Vendored build but smoke failed: " << err;
    EXPECT_TRUE(err.empty());
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_simple_pir_real=1"), std::string::npos)
        << "Stub error must guide callers to the activation flag; got: "
        << err;
  }
}

TEST(SimplePirRuntimeTest, SmokeIsIdempotent) {
  std::string err1, err2;
  auto rc1 = SimplePirRuntime::Instance().SmokeMatMul(&err1);
  auto rc2 = SimplePirRuntime::Instance().SmokeMatMul(&err2);
  EXPECT_EQ(rc1, rc2);
  EXPECT_EQ(err1, err2);
}

}  // namespace
}  // namespace primihub::pir::simple_pir
