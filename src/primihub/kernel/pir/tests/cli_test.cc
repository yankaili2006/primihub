/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * cli_test — covers the constraint-resolution + selector behaviours that
 * `pir_inspect auto` exercises. The CLI itself is a thin wrapper around
 * PirRegistry + PirSelector; the value of this test is asserting the
 * happy-path scenarios documented in `pir_inspect --help` actually
 * resolve as the docs claim. argv parsing in pir_inspect_main.cc is
 * out of scope (covered by manual smoke).
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/selector.h"

namespace primihub::pir {
namespace {

class StubOperator : public BasePirOperator {
 public:
  explicit StubOperator(const Options& options) : BasePirOperator(options) {}
  retcode OnExecute(const PirDataType&, PirDataType*) override {
    return retcode::SUCCESS;
  }
};

// Register stub algorithms covering the perf-class spread that the CLI
// docs reference, so the assertions below do not depend on which real
// algorithms happened to be linked into the test binary.
void RegisterStub(const std::string& name, const PirCapabilities& caps) {
  PirRegistry::Instance().Register(
      name,
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new StubOperator(o));
      },
      caps);
}

void EnsureStubsRegistered() {
  PirRegistry::EnsureRegistered();

  // A two-server, ms-latency option (DoublePIR profile).
  PirCapabilities two_server_ms;
  two_server_ms.query_types = {QueryType::Index};
  two_server_ms.min_servers = 2;
  two_server_ms.max_servers = 2;
  two_server_ms.backends = {Backend::CPU};
  two_server_ms.perf_class = PerfClass::Ms;
  two_server_ms.threat_model = ThreatModel::SemiHonestNonColluding;
  two_server_ms.recommended_max_db_size = 1ull << 30;
  two_server_ms.typical_query_comm_bytes = 256ull * 1024;
  two_server_ms.is_real = true;  // stub simulates a real algo
  RegisterStub("clit_two_server_ms", two_server_ms);

  // A single-server, seconds-class option (SealPIR/Spiral profile).
  PirCapabilities single_seconds;
  single_seconds.query_types = {QueryType::Index};
  single_seconds.min_servers = 1;
  single_seconds.max_servers = 1;
  single_seconds.backends = {Backend::CPU};
  single_seconds.perf_class = PerfClass::Seconds;
  single_seconds.threat_model = ThreatModel::SemiHonest;
  single_seconds.recommended_max_db_size = 1ull << 28;
  single_seconds.typical_query_comm_bytes = 32ull * 1024;
  single_seconds.is_real = true;
  RegisterStub("clit_single_seconds", single_seconds);

  // A keyword-only option (APSI profile).
  PirCapabilities keyword_caps;
  keyword_caps.query_types = {QueryType::Keyword};
  keyword_caps.min_servers = 1;
  keyword_caps.max_servers = 1;
  keyword_caps.backends = {Backend::CPU};
  keyword_caps.perf_class = PerfClass::SubSecond;
  keyword_caps.threat_model = ThreatModel::SemiHonest;
  keyword_caps.recommended_max_db_size = 1ull << 28;
  keyword_caps.typical_query_comm_bytes = 16ull * 1024;
  keyword_caps.is_real = true;
  RegisterStub("clit_keyword", keyword_caps);
}

bool ContainsAlgorithm(const std::vector<std::string>& names,
                       const std::string& target) {
  return std::find(names.begin(), names.end(), target) != names.end();
}

const AlgoMatch* FindMatch(const std::vector<AlgoMatch>& matches,
                           const std::string& target) {
  for (const auto& m : matches) {
    if (m.algorithm == target) return &m;
  }
  return nullptr;
}

TEST(PirCliResolutionTest, ListMatchesRegistryEnumeration) {
  // `pir_inspect list` returns whatever the registry knows about; we just
  // require the stubs we registered show up.
  EnsureStubsRegistered();
  auto names = PirRegistry::Instance().ListAlgorithms();
  EXPECT_TRUE(ContainsAlgorithm(names, "clit_two_server_ms"));
  EXPECT_TRUE(ContainsAlgorithm(names, "clit_single_seconds"));
  EXPECT_TRUE(ContainsAlgorithm(names, "clit_keyword"));
}

TEST(PirCliResolutionTest, CapsExposesJsonForRegisteredAlgo) {
  // `pir_inspect caps <name>` dumps GetCapabilities()->ToJson().
  EnsureStubsRegistered();
  const auto* caps =
      PirRegistry::Instance().GetCapabilities("clit_two_server_ms");
  ASSERT_NE(caps, nullptr);
  const auto json = caps->ToJson();
  EXPECT_NE(json.find("min_servers"), std::string::npos);
  EXPECT_NE(json.find("perf_class"), std::string::npos);
}

TEST(PirCliResolutionTest, CapsReturnsNullForUnknown) {
  // CLI prints "algorithm not registered" and exits 1 — the null return
  // value is the signal it relies on.
  const auto* caps = PirRegistry::Instance().GetCapabilities("__no_such__");
  EXPECT_EQ(caps, nullptr);
}

TEST(PirCliResolutionTest, AutoMsBudgetPicksTwoServerWhenAllowed) {
  // `pir_inspect auto db-size=1e8 query-type=index allow-two-server=true
  //  assume-non-colluding=true latency-budget=ms` — should rank the
  // two-server stub above the seconds-class single-server stub.
  EnsureStubsRegistered();
  Constraints c;
  c.db_size = static_cast<uint64_t>(1e8);
  c.query_type = QueryType::Index;
  c.allow_two_server = true;
  c.assume_non_colluding = true;
  c.latency_budget = LatencyBudget::Ms;
  c.min_threat_model = ThreatModel::SemiHonest;

  auto matches = PirSelector{}.RecommendWithRationale(c);
  const auto* two_server = FindMatch(matches, "clit_two_server_ms");
  const auto* single = FindMatch(matches, "clit_single_seconds");
  ASSERT_NE(two_server, nullptr);
  ASSERT_NE(single, nullptr);
  // The two-server algo must pass under ms budget; the single-server
  // seconds-class algo must NOT pass under a ms budget.
  EXPECT_TRUE(two_server->passes);
  EXPECT_FALSE(single->passes);
}

TEST(PirCliResolutionTest, AutoExcludesTwoServerWithoutOptIn) {
  // Even though the two-server stub could satisfy the latency budget, the
  // selector must NOT recommend it unless allow_two_server AND
  // assume_non_colluding are both set — otherwise the threat model is wrong.
  EnsureStubsRegistered();
  Constraints c;
  c.db_size = static_cast<uint64_t>(1e8);
  c.query_type = QueryType::Index;
  c.allow_two_server = false;  // <-- no opt-in
  c.assume_non_colluding = false;
  c.latency_budget = LatencyBudget::Ms;

  auto matches = PirSelector{}.RecommendWithRationale(c);
  const auto* two_server = FindMatch(matches, "clit_two_server_ms");
  ASSERT_NE(two_server, nullptr);
  EXPECT_FALSE(two_server->passes);
  EXPECT_FALSE(two_server->failed_checks.empty())
      << "Expected the dry-run rationale to explain WHY two-server algo was "
         "rejected (allow_two_server=false).";
}

TEST(PirCliResolutionTest, KeywordQueryTypeRoutesToKeywordCapableAlgo) {
  // `pir_inspect auto query-type=keyword` must skip index-only algorithms.
  EnsureStubsRegistered();
  Constraints c;
  c.db_size = static_cast<uint64_t>(1e6);
  c.query_type = QueryType::Keyword;
  c.latency_budget = LatencyBudget::SubSecond;

  auto matches = PirSelector{}.RecommendWithRationale(c);
  const auto* kw = FindMatch(matches, "clit_keyword");
  const auto* idx = FindMatch(matches, "clit_single_seconds");
  ASSERT_NE(kw, nullptr);
  ASSERT_NE(idx, nullptr);
  EXPECT_TRUE(kw->passes);
  EXPECT_FALSE(idx->passes)
      << "Index-only algorithm must be filtered for Keyword query.";
}

TEST(PirCliResolutionTest, RecommendReturnsPassesOnlyOrdered) {
  // The non-dry-run path: Recommend returns just the passing names. The
  // CLI prints the first one and exits.
  EnsureStubsRegistered();
  Constraints c;
  c.db_size = static_cast<uint64_t>(1e6);
  c.query_type = QueryType::Index;
  c.allow_two_server = true;
  c.assume_non_colluding = true;
  c.latency_budget = LatencyBudget::Ms;

  auto ranked = PirSelector{}.Recommend(c);
  ASSERT_FALSE(ranked.empty());
  // None of the entries should be one we know fails for this constraint.
  for (const auto& name : ranked) {
    EXPECT_NE(name, "clit_keyword")
        << "Recommend must not return Keyword-only algos for Index queries.";
  }
}

}  // namespace
}  // namespace primihub::pir
