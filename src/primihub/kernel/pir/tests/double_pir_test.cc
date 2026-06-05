/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Skeleton-phase test for DoublePirOperator. Verifies registration,
 * capability surface, factory creation, OnExecute failure-mode, and the
 * key selector behaviour that distinguishes DoublePIR from single-server
 * algorithms: it MUST be filtered out unless the caller declares both
 * allow_two_server and assume_non_colluding.
 *
 * The cryptographic core (tasks 5.4-5.10) is not yet wired; this file
 * exercises only the framework plumbing.
 */
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/selector.h"

namespace primihub::pir {

class DoublePirSkeletonTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PirRegistry::EnsureRegistered();
  }
};

TEST_F(DoublePirSkeletonTest, RegisteredUnderDoublePirName) {
  auto algos = PirRegistry::Instance().ListAlgorithms();
  bool found = false;
  for (const auto& a : algos) {
    if (a == "double_pir") { found = true; break; }
  }
  EXPECT_TRUE(found);
}

TEST_F(DoublePirSkeletonTest, CapabilityProfileMatchesPaper) {
  const auto* caps = PirRegistry::Instance().GetCapabilities("double_pir");
  ASSERT_NE(caps, nullptr);
  // From Henzinger et al., USENIX'23.
  EXPECT_EQ(caps->query_types.count(QueryType::Index), 1u);
  EXPECT_EQ(caps->min_servers, 2u);
  EXPECT_EQ(caps->max_servers, 2u);  // Exactly two, not "at least two".
  EXPECT_TRUE(caps->needs_preprocess);
  EXPECT_TRUE(caps->hint_per_database);
  EXPECT_EQ(caps->threat_model, ThreatModel::SemiHonestNonColluding);
  EXPECT_EQ(caps->perf_class, PerfClass::Ms);
  EXPECT_GE(caps->recommended_max_db_size, 100'000'000ULL);  // >= 1e8
  EXPECT_GT(caps->typical_query_comm_bytes, 0u);
  // Hint is the algorithm's primary cost; must be reflected as nonzero.
  EXPECT_GT(caps->typical_hint_size_bytes, 0u);
  EXPECT_EQ(caps->Check(), "");
}

TEST_F(DoublePirSkeletonTest, FactoryReturnsNonNull) {
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  auto op = PirRegistry::Instance().Create("double_pir", opt);
  ASSERT_NE(op, nullptr);
}

TEST_F(DoublePirSkeletonTest, OnExecuteFailsLoudlyUntilImplemented) {
  EXPECT_TRUE(DoublePirOperator::kIsSkeleton);
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  DoublePirOperator op(opt);
  PirDataType input;
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
  EXPECT_TRUE(result.empty());
}

// The single most important behavioural test for DoublePIR — it MUST be
// filtered out unless the caller declares both allow_two_server AND
// assume_non_colluding. Catching this regression early avoids the worst
// possible outcome: silently routing a query intended for a single-trust
// model to a two-server protocol that requires non-collusion.
TEST_F(DoublePirSkeletonTest, SelectorFiltersWithoutNonColludingAssumption) {
  Constraints c;
  c.db_size = 100'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Ms;
  c.allow_two_server = true;
  c.assume_non_colluding = false;   // <- the key flag
  auto matches = PirSelector{}.RecommendWithRationale(c);
  for (const auto& m : matches) {
    if (m.algorithm == "double_pir") {
      EXPECT_FALSE(m.passes)
          << "DoublePIR passed selector without non-colluding assumption";
      return;
    }
  }
  FAIL() << "DoublePIR not present in selector output at all";
}

TEST_F(DoublePirSkeletonTest, SelectorFiltersWithoutAllowTwoServer) {
  Constraints c;
  c.db_size = 100'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Ms;
  c.allow_two_server = false;      // <- the key flag
  c.assume_non_colluding = true;
  auto matches = PirSelector{}.RecommendWithRationale(c);
  for (const auto& m : matches) {
    if (m.algorithm == "double_pir") {
      EXPECT_FALSE(m.passes)
          << "DoublePIR passed selector without allow_two_server";
      return;
    }
  }
  FAIL() << "DoublePIR not present in selector output at all";
}

TEST_F(DoublePirSkeletonTest, SelectorPicksDoublePirWhenAllConditionsMet) {
  Constraints c;
  c.db_size = 100'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Ms;
  c.allow_two_server = true;
  c.assume_non_colluding = true;
  c.client_can_cache_hint = true;
  auto matches = PirSelector{}.RecommendWithRationale(c);
  bool double_pir_passes = false;
  for (const auto& m : matches) {
    if (m.algorithm == "double_pir") {
      double_pir_passes = m.passes;
      break;
    }
  }
  EXPECT_TRUE(double_pir_passes)
      << "DoublePIR did not pass selector despite all conditions met "
         "(db_size=1e8, Ms budget, two-server allowed, non-colluding "
         "assumed) — selector or capability profile regressed";
}

}  // namespace primihub::pir
