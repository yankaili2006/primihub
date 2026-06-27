/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tiptoe operator test (task 1.1). Mode-aware: the default .50 build has no SEAL
 * so the operator is a skeleton (OnExecute -> FAIL, is_real=false); built with
 * --define=enable_tiptoe_real=1 it is a real operator and the end-to-end
 * retrieval branch runs (validated standalone against SEAL 4.1 when 1.1f
 * landed). The capability shape + registration are checked in both modes.
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_pir.h"

#include <algorithm>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED
#include <cstdint>
#include "base64.h"  // NOLINT
#endif

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

TEST(TiptoePirTest, CapabilityShape) {
  PirRegistry::EnsureRegistered();
  const PirCapabilities* caps =
      PirRegistry::Instance().GetCapabilities("tiptoe");
  ASSERT_NE(caps, nullptr);
  EXPECT_EQ(caps->min_servers, 1u);
  EXPECT_EQ(caps->max_servers, 1u);
  EXPECT_TRUE(caps->needs_preprocess);
  EXPECT_TRUE(caps->hint_per_database);
  EXPECT_EQ(caps->query_types.count(QueryType::Semantic), 1u);
}

TEST(TiptoePirTest, RealnessMatchesBuildMode) {
  PirRegistry::EnsureRegistered();
  const PirCapabilities* caps =
      PirRegistry::Instance().GetCapabilities("tiptoe");
  ASSERT_NE(caps, nullptr);
#ifdef PIR_TIPTOE_RLWE_VENDORED
  EXPECT_TRUE(caps->is_real);
  EXPECT_FALSE(TiptoePirOperator::kIsSkeleton);
#else
  EXPECT_FALSE(caps->is_real);
  EXPECT_TRUE(TiptoePirOperator::kIsSkeleton);
#endif
}

#ifdef PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoePirTest, OnExecuteRetrievesBytes) {
  // 9 single-byte elements -> 3x3 DB.
  std::vector<std::string> db;
  for (int i = 0; i < 9; ++i) {
    const std::uint8_t b = static_cast<std::uint8_t>(i * 28 + 7);
    db.push_back(base64_encode(&b, 1));
  }
  TiptoePirOperator op(MakeOpts());
  PirDataType in;
  in["db_content"] = db;
  in["query_indices"] = {"0", "4", "8"};
  PirDataType out;
  ASSERT_EQ(op.OnExecute(in, &out), retcode::SUCCESS);
  ASSERT_EQ(out["recovered"].size(), 3u);
  EXPECT_EQ(out["recovered"][0], db[0]);
  EXPECT_EQ(out["recovered"][1], db[4]);
  EXPECT_EQ(out["recovered"][2], db[8]);
}

TEST(TiptoePirTest, RejectsMultiByteElements) {
  TiptoePirOperator op(MakeOpts());
  PirDataType in;
  std::vector<std::uint8_t> two = {1, 2};
  in["db_content"] = {base64_encode(two.data(), two.size())};
  in["query_indices"] = {"0"};
  PirDataType out;
  EXPECT_EQ(op.OnExecute(in, &out), retcode::FAIL);
}

#else  // skeleton mode

TEST(TiptoePirTest, OnExecuteSkeletonReturnsFail) {
  TiptoePirOperator op(MakeOpts());
  PirDataType in;
  in["db_content"] = {"AA=="};
  in["query_indices"] = {"0"};
  PirDataType out;
  EXPECT_EQ(op.OnExecute(in, &out), retcode::FAIL);
}

#endif

}  // namespace
}  // namespace primihub::pir
