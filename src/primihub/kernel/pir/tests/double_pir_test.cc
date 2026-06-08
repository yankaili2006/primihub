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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <vector>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_cache.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
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


// ---- Operator integration tests (chunk 7 of openspec task 5.5) ----
//
// kIsSkeleton flipped to false in chunk 7. DoublePirOperator now
// drives the full Init/Setup/Query/Answer/Recover pipeline on the
// caller-supplied PirDataType input.

TEST(DoublePirOperatorTest, NotSkeletonAfterChunk7) {
  EXPECT_FALSE(DoublePirOperator::kIsSkeleton);
}

TEST(DoublePirOperatorTest, FailsOnMissingDb) {
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  DoublePirOperator op(opt);
  PirDataType input;
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(DoublePirOperatorTest, FailsOnMissingIndices) {
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = {"1", "2", "3"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(DoublePirOperatorTest, EndToEndRetrievesCorrectEntries) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "end-to-end retrieval needs the kernel bridge";
  }
  // 64-entry DB. l=8, m=8 — l divisible by 8 (kRowAlignment) and
  // info.x=1 from SetupShape with p_double=929, row_length=8 →
  // params.l % info.x == 0.
  std::vector<std::string> db;
  db.reserve(64);
  for (uint64_t i = 0; i < 64; ++i) {
    db.push_back(std::to_string((i * 13 + 7) & 0xFF));
  }
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"0", "27", "63"};

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  auto it = result.find("recovered");
  ASSERT_NE(it, result.end());
  ASSERT_EQ(it->second.size(), 3u);
  EXPECT_EQ(it->second[0], std::to_string((0u * 13 + 7) & 0xFF));
  EXPECT_EQ(it->second[1], std::to_string((27u * 13 + 7) & 0xFF));
  EXPECT_EQ(it->second[2], std::to_string((63u * 13 + 7) & 0xFF));
}

// --- task 5.6 chunk 5: operator-level HintCache persistence ----------

namespace {

// RAII tmpfile path used by the persistence integration tests.
class TmpPath {
 public:
  TmpPath() {
    char tmpl[] = "/tmp/double_pir_persist_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd >= 0) {
      ::close(fd);
      path_ = tmpl;
      // Persistence test wants to control file lifetime — unlink the
      // mkstemp result so the first SaveToFile gets a clean create.
      ::unlink(path_.c_str());
    }
  }
  ~TmpPath() {
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

void RunOnceWithHintPath(const std::string& hint_path) {
  std::vector<std::string> db;
  db.reserve(64);
  for (uint64_t i = 0; i < 64; ++i) {
    db.push_back(std::to_string((i * 13 + 7) & 0xFF));
  }
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  opt.hint_path = hint_path;
  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"0", "27", "63"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
}

}  // namespace

TEST(DoublePirOperatorPersistTest, PersistsHintCacheAcrossInstances) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());

  // Cold cache, no file. First run: miss + save.
  double_pir::HintCache::Instance().Clear();
  const uint64_t hits0 = double_pir::HintCache::Instance().Hits();
  const uint64_t misses0 = double_pir::HintCache::Instance().Misses();
  RunOnceWithHintPath(tmp.path());
  EXPECT_EQ(double_pir::HintCache::Instance().Hits(), hits0);
  EXPECT_EQ(double_pir::HintCache::Instance().Misses(), misses0 + 1u);
  // File should now exist with at least one entry.
  std::ifstream check(tmp.path(), std::ios::binary);
  ASSERT_TRUE(check.is_open()) << "SaveToFile did not create " << tmp.path();
  check.close();

  // Wipe singleton (Clear() also resets loaded_paths_).
  double_pir::HintCache::Instance().Clear();
  const uint64_t hits1 = double_pir::HintCache::Instance().Hits();
  const uint64_t misses1 = double_pir::HintCache::Instance().Misses();

  // Second run with same hint_path → MaybeLoadOnce loads, then TryGet
  // hits (same db + params hash to the same fingerprint).
  RunOnceWithHintPath(tmp.path());
  EXPECT_EQ(double_pir::HintCache::Instance().Hits(), hits1 + 1u);
  EXPECT_EQ(double_pir::HintCache::Instance().Misses(), misses1);
}

TEST(DoublePirOperatorPersistTest, MissingHintFileDegradesGracefully) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());
  // tmp.path() does NOT exist yet (TmpPath unlinked it on creation).
  double_pir::HintCache::Instance().Clear();
  RunOnceWithHintPath(tmp.path());
  // Operator must succeed even though MaybeLoadOnce hit open() failure.
  // SaveToFile after the miss creates the file.
  std::ifstream check(tmp.path(), std::ios::binary);
  EXPECT_TRUE(check.is_open());
}

TEST(DoublePirOperatorPersistTest, CorruptHintFileDegradesGracefully) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());
  // Pre-populate with garbage. MaybeLoadOnce will fail (bad magic) but
  // operator must still finish; SaveToFile then overwrites with a valid
  // cache file.
  {
    std::ofstream out(tmp.path(), std::ios::binary | std::ios::trunc);
    const char garbage[] = "definitely not a PHHC cache file XXXXXX";
    out.write(garbage, sizeof(garbage) - 1);
  }
  double_pir::HintCache::Instance().Clear();
  RunOnceWithHintPath(tmp.path());

  // After the second run with same path the corrupt file has been
  // overwritten — a fresh load should succeed.
  double_pir::HintCache::Instance().Clear();
  std::string err;
  EXPECT_EQ(double_pir::HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS) << err;
  EXPECT_GE(double_pir::HintCache::Instance().Size(), 1u);
}

}  // namespace primihub::pir
