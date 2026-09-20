/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/selector.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

class StubOp : public BasePirOperator {
 public:
  explicit StubOp(const Options& o) : BasePirOperator(o) {}
  retcode OnExecute(const PirDataType&, PirDataType*) override {
    return retcode::SUCCESS;
  }
};

// Helper to register a fake algorithm with chosen caps; each test uses
// unique algorithm names so registrations from prior tests do not interfere.
void RegisterFake(const std::string& name, const PirCapabilities& caps) {
  PirRegistry::Instance().Register(
      name,
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new StubOp(o));
      },
      caps);
}

PirCapabilities SpiralLike() {
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 1;
  c.max_servers = 1;
  c.needs_preprocess = false;
  c.hint_per_database = false;
  c.threat_model = ThreatModel::SemiHonest;
  c.perf_class = PerfClass::Seconds;
  c.recommended_max_db_size = 100000000ULL;
  c.backends = {Backend::CPU, Backend::AVX2};
  c.typical_query_comm_bytes = 14336;
  c.typical_hint_size_bytes = 0;
  c.is_real = true;  // test stubs simulate real algos for selector logic
  return c;
}

PirCapabilities DoubleLike() {
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 2;
  c.max_servers = 2;
  c.needs_preprocess = true;
  c.hint_per_database = true;  // hint shared across clients
  c.threat_model = ThreatModel::SemiHonestNonColluding;
  c.perf_class = PerfClass::Ms;
  c.recommended_max_db_size = 10000000000ULL;
  c.backends = {Backend::CPU, Backend::AVX2};
  c.typical_query_comm_bytes = 4096;
  c.typical_hint_size_bytes = 200000000ULL;
  c.is_real = true;
  return c;
}

PirCapabilities ApsiLike() {
  PirCapabilities c;
  c.query_types = {QueryType::Keyword};
  c.min_servers = 1;
  c.max_servers = 1;
  c.needs_preprocess = true;
  c.hint_per_database = true;
  c.threat_model = ThreatModel::SemiHonest;
  c.perf_class = PerfClass::Seconds;
  c.recommended_max_db_size = 10000000ULL;
  c.backends = {Backend::CPU};
  c.typical_query_comm_bytes = 1048576;
  c.typical_hint_size_bytes = 500000000ULL;
  c.is_real = true;
  return c;
}

TEST(PirSelectorTest, IndexQueryWithoutTwoServerExcludesDoublePir) {
  RegisterFake("sel_test_spiral_a", SpiralLike());
  RegisterFake("sel_test_double_a", DoubleLike());

  Constraints c;
  c.db_size = 100000000ULL;
  c.query_type = QueryType::Index;
  c.allow_two_server = false;

  auto algos = PirSelector{}.Recommend(c);
  EXPECT_NE(std::find(algos.begin(), algos.end(), "sel_test_spiral_a"),
            algos.end());
  EXPECT_EQ(std::find(algos.begin(), algos.end(), "sel_test_double_a"),
            algos.end()) << "double_pir must be excluded when allow_two_server=false";
}

TEST(PirSelectorTest, DoublePirRequiresNonColludingAssumption) {
  RegisterFake("sel_test_double_b", DoubleLike());

  Constraints c;
  c.db_size = 100000000ULL;
  c.query_type = QueryType::Index;
  c.allow_two_server = true;
  c.assume_non_colluding = false;  // not declared

  auto algos = PirSelector{}.Recommend(c);
  EXPECT_EQ(std::find(algos.begin(), algos.end(), "sel_test_double_b"),
            algos.end()) << "double_pir must be excluded without non-colluding";
}

TEST(PirSelectorTest, TwoServerNonColludingAtScaleDoubleWinsOverSpiral) {
  RegisterFake("sel_test_spiral_c", SpiralLike());
  RegisterFake("sel_test_double_c", DoubleLike());

  Constraints c;
  c.db_size = 100000000ULL;
  c.query_type = QueryType::Index;
  c.allow_two_server = true;
  c.assume_non_colluding = true;
  c.latency_budget = LatencyBudget::Seconds;
  c.client_can_cache_hint = true;

  auto algos = PirSelector{}.Recommend(c);
  ASSERT_FALSE(algos.empty());
  // Registry singleton spans test cases, so other fixtures may appear too.
  // Validate the relative ordering of our two registered names only.
  int pos_double = -1, pos_spiral = -1;
  for (int i = 0; i < static_cast<int>(algos.size()); ++i) {
    if (algos[i] == "sel_test_double_c") pos_double = i;
    if (algos[i] == "sel_test_spiral_c") pos_spiral = i;
  }
  ASSERT_GE(pos_double, 0) << "double_c must appear in recommendations";
  ASSERT_GE(pos_spiral, 0) << "spiral_c must appear in recommendations";
  EXPECT_LT(pos_double, pos_spiral)
      << "double_pir (Ms class) must rank before spiral (Seconds class)";
}

TEST(PirSelectorTest, KeywordQueryPicksApsi) {
  RegisterFake("sel_test_apsi_d", ApsiLike());
  RegisterFake("sel_test_spiral_d", SpiralLike());

  Constraints c;
  c.db_size = 1000000ULL;
  c.query_type = QueryType::Keyword;
  c.client_can_cache_hint = true;

  auto algos = PirSelector{}.Recommend(c);
  ASSERT_FALSE(algos.empty());
  EXPECT_NE(std::find(algos.begin(), algos.end(), "sel_test_apsi_d"),
            algos.end());
  EXPECT_EQ(std::find(algos.begin(), algos.end(), "sel_test_spiral_d"),
            algos.end()) << "Spiral only does Index; must be excluded";
}

TEST(PirSelectorTest, MsBudgetExcludesSlowAlgos) {
  RegisterFake("sel_test_spiral_e", SpiralLike());  // perf=Seconds
  RegisterFake("sel_test_double_e", DoubleLike());  // perf=Ms

  Constraints c;
  c.db_size = 100000000ULL;
  c.query_type = QueryType::Index;
  c.allow_two_server = true;
  c.assume_non_colluding = true;
  c.latency_budget = LatencyBudget::Ms;
  c.client_can_cache_hint = true;

  auto algos = PirSelector{}.Recommend(c);
  EXPECT_EQ(std::find(algos.begin(), algos.end(), "sel_test_spiral_e"),
            algos.end()) << "Seconds class fails Ms budget";
  EXPECT_NE(std::find(algos.begin(), algos.end(), "sel_test_double_e"),
            algos.end());
}

TEST(PirSelectorTest, UnsatisfiableReturnsEmpty) {
  RegisterFake("sel_test_unreach", SpiralLike());

  Constraints c;
  c.query_type = QueryType::Semantic;  // nothing registered supports Semantic
  c.allow_two_server = false;
  c.min_threat_model = ThreatModel::Malicious;  // nothing supports Malicious

  auto algos = PirSelector{}.Recommend(c);
  EXPECT_TRUE(algos.empty());
}

TEST(PirSelectorTest, RationaleReportsFailureReasons) {
  RegisterFake("sel_test_rat", DoubleLike());

  Constraints c;
  c.query_type = QueryType::Index;
  c.allow_two_server = false;  // will cause failure
  c.assume_non_colluding = false;  // will also cause failure

  auto matches = PirSelector{}.RecommendWithRationale(c);
  // Find the failing match for sel_test_rat
  bool found = false;
  for (const auto& m : matches) {
    if (m.algorithm == "sel_test_rat") {
      EXPECT_FALSE(m.passes);
      EXPECT_FALSE(m.failed_checks.empty());
      found = true;
    }
  }
  EXPECT_TRUE(found);
}


// Regression coverage for the hint-cacheability predicate bug fix.
// Before the fix, the selector filtered per-client hint algorithms (hint
// size > 0 AND hint_per_database == false), which is the inverted
// semantic — per-client hints are generated fresh per query and can never
// be cached. The real cacheability question is about per-database hints
// (shared across queries against the same DB), so an honest
// client_can_cache_hint=false caller MUST see those filtered out.
TEST(PirSelectorTest, PerDatabaseHintFilteredWhenClientCannotCache) {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 1'000'000'000ULL;
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 1024;
  caps.typical_hint_size_bytes = 16ULL * 1024 * 1024;
  caps.is_real = true;
  RegisterFake("sel_test_hint_per_db", caps);

  Constraints c;
  c.db_size = 100'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Ms;
  c.client_can_cache_hint = false;

  auto matches = PirSelector{}.RecommendWithRationale(c);
  bool found_failing = false;
  for (const auto& m : matches) {
    if (m.algorithm == "sel_test_hint_per_db") {
      EXPECT_FALSE(m.passes);
      bool has_cache_failure = false;
      for (const auto& f : m.failed_checks) {
        if (f.find("cache") != std::string::npos) has_cache_failure = true;
      }
      EXPECT_TRUE(has_cache_failure)
          << "Filter must cite the cacheability constraint";
      found_failing = true;
    }
  }
  EXPECT_TRUE(found_failing);
}

// Skeleton-filter: a cap with is_real=false must be excluded from
// Recommend() by default, but appear in RecommendWithRationale() with
// a "skeleton" fail reason — and become a passing candidate when the
// caller sets include_skeletons=true.
TEST(PirSelectorTest, SkeletonExcludedByDefault) {
  // Two fakes: one real, one skeleton, otherwise identical caps.
  PirCapabilities real_caps = SpiralLike();  // is_real=true
  PirCapabilities skel_caps = SpiralLike();
  skel_caps.is_real = false;
  RegisterFake("sel_test_skel_real", real_caps);
  RegisterFake("sel_test_skel_stub", skel_caps);

  Constraints c;
  c.db_size = 1'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Any;

  // Default: skeleton excluded from passing recommendations.
  auto names = PirSelector{}.Recommend(c);
  EXPECT_TRUE(std::find(names.begin(), names.end(), "sel_test_skel_real")
              != names.end());
  EXPECT_TRUE(std::find(names.begin(), names.end(), "sel_test_skel_stub")
              == names.end())
      << "skeleton should NOT appear in default Recommend()";

  // Dry-run table includes the skeleton with the activation hint reason.
  auto matches = PirSelector{}.RecommendWithRationale(c);
  bool found_skel_with_reason = false;
  for (const auto& m : matches) {
    if (m.algorithm == "sel_test_skel_stub") {
      EXPECT_FALSE(m.passes);
      for (const auto& f : m.failed_checks) {
        if (f.find("skeleton") != std::string::npos) {
          found_skel_with_reason = true;
        }
      }
    }
  }
  EXPECT_TRUE(found_skel_with_reason)
      << "skeleton must surface with a 'skeleton — set include_skeletons'"
         " message in the dry-run table";
}

TEST(PirSelectorTest, IncludeSkeletonsOptsIn) {
  PirCapabilities skel_caps = SpiralLike();
  skel_caps.is_real = false;
  RegisterFake("sel_test_optin_skel", skel_caps);

  Constraints c;
  c.db_size = 1'000'000ULL;
  c.query_type = QueryType::Index;
  c.latency_budget = LatencyBudget::Any;
  c.include_skeletons = true;

  auto names = PirSelector{}.Recommend(c);
  EXPECT_TRUE(std::find(names.begin(), names.end(), "sel_test_optin_skel")
              != names.end())
      << "include_skeletons=true should bring the skeleton back as a"
         " passing candidate";
}

}  // namespace
}  // namespace primihub::pir
