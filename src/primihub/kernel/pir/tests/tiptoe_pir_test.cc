/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tiptoe skeleton + registry test (task 1.1a). Pins that the operator is
 * registered under "tiptoe", carries the canonical capability profile (still a
 * skeleton: is_real=false), and that OnExecute returns FAIL until the real
 * BFV-on-SimplePIR path lands (chunks 1.1b-1.1f). These expectations flip in
 * chunk 1.1f when the operator becomes real.
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_pir.h"

#include <algorithm>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

Options MakeOpts() {
  Options o;
  o.code = "tiptoe_pir_test";
  o.role = Role::CLIENT;
  return o;
}

TEST(TiptoePirTest, RegisteredUnderTiptoeName) {
  PirRegistry::EnsureRegistered();
  const std::vector<std::string> algos =
      PirRegistry::Instance().ListAlgorithms();
  EXPECT_NE(std::find(algos.begin(), algos.end(), "tiptoe"), algos.end());
}

TEST(TiptoePirTest, CapabilityProfileMatchesSpec) {
  PirRegistry::EnsureRegistered();
  const PirCapabilities* caps =
      PirRegistry::Instance().GetCapabilities("tiptoe");
  ASSERT_NE(caps, nullptr);
  EXPECT_FALSE(caps->is_real);  // skeleton until chunk 1.1f
  EXPECT_EQ(caps->min_servers, 1u);
  EXPECT_EQ(caps->max_servers, 1u);
  EXPECT_TRUE(caps->needs_preprocess);
  EXPECT_TRUE(caps->hint_per_database);
  EXPECT_EQ(caps->query_types.count(QueryType::Semantic), 1u);
}

TEST(TiptoePirTest, IsSkeleton) {
  EXPECT_TRUE(TiptoePirOperator::kIsSkeleton);
}

TEST(TiptoePirTest, OnExecuteSkeletonReturnsFail) {
  TiptoePirOperator op(MakeOpts());
  PirDataType in;
  in["db_content"] = {"AA=="};
  in["query_indices"] = {"0"};
  PirDataType out;
  EXPECT_EQ(op.OnExecute(in, &out), retcode::FAIL);
}

}  // namespace
}  // namespace primihub::pir
