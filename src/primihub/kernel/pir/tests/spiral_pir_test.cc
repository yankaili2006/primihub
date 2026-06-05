/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Skeleton-phase test for SpiralPirOperator. These tests verify the framework
 * plumbing — registration, capability surface, factory creation, OnExecute
 * failure-mode — and intentionally do NOT exercise the cryptographic core,
 * which is not yet wired (see openspec/changes/primihub-pir-multi-algo tasks
 * 4.1/4.2/4.5).
 *
 * Once the real implementation lands, drop the SkeletonBehavior test cases
 * (or rewrite them to exercise the real query path) and add the
 * correctness suite scoped by task 4.7 (1e3/1e5/1e7 roundtrip).
 */
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_pir.h"

namespace primihub::pir {

class SpiralPirSkeletonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // EnsureRegistered drains the static-init pending queue into the real
    // registry; safe to call repeatedly across tests because the pending
    // queue has already been emptied after the first call.
    PirRegistry::EnsureRegistered();
  }
};

TEST_F(SpiralPirSkeletonTest, RegisteredUnderSpiralName) {
  auto algos = PirRegistry::Instance().ListAlgorithms();
  bool found = false;
  for (const auto& a : algos) {
    if (a == "spiral") { found = true; break; }
  }
  EXPECT_TRUE(found) << "spiral not in ListAlgorithms; check that "
                        "spiral_pir_operator BUILD target is in test deps "
                        "(alwayslink) and registrar TU is being linked.";
}

TEST_F(SpiralPirSkeletonTest, CapabilityProfileMatchesPaper) {
  const auto* caps = PirRegistry::Instance().GetCapabilities("spiral");
  ASSERT_NE(caps, nullptr);
  // From Menon & Wu, USENIX'22 §6 + paper claims.
  EXPECT_EQ(caps->query_types.size(), 1u);
  EXPECT_EQ(caps->query_types.count(QueryType::Index), 1u);
  EXPECT_EQ(caps->min_servers, 1u);
  EXPECT_EQ(caps->max_servers, 1u);
  EXPECT_FALSE(caps->needs_preprocess);
  EXPECT_FALSE(caps->hint_per_database);
  EXPECT_EQ(caps->threat_model, ThreatModel::SemiHonest);
  EXPECT_EQ(caps->perf_class, PerfClass::Seconds);
  EXPECT_GE(caps->recommended_max_db_size, 100'000'000ULL);  // >= 1e8
  EXPECT_EQ(caps->backends.size(), 1u);
  EXPECT_EQ(caps->backends.count(Backend::CPU), 1u);
  EXPECT_GT(caps->typical_query_comm_bytes, 0u);
  EXPECT_EQ(caps->typical_hint_size_bytes, 0u);
  EXPECT_EQ(caps->Check(), "");
}

TEST_F(SpiralPirSkeletonTest, FactoryReturnsNonNull) {
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  auto op = PirRegistry::Instance().Create("spiral", opt);
  ASSERT_NE(op, nullptr) << "Create('spiral', opts) returned nullptr — "
                           "registrar or creator lambda mis-wired.";
}

TEST_F(SpiralPirSkeletonTest, OnExecuteFailsLoudlyUntilImplemented) {
  // The skeleton is a deliberate stub: silent wrong answers are worse than
  // loud failure. When the real implementation lands, replace this test with
  // a correctness check.
  EXPECT_TRUE(SpiralPirOperator::kIsSkeleton)
      << "kIsSkeleton flipped to false — implementation may have landed; "
         "rewrite this test against the real query path.";
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  SpiralPirOperator op(opt);
  PirDataType input;
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
  EXPECT_TRUE(result.empty());
}

}  // namespace primihub::pir
