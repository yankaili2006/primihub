/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirOperator OnExecute integration tests. Validates the
 * single-process self-contained retrieval path: caller passes a DB
 * + indices in the PirDataType map; operator runs the full
 * SimplePirProtocol pipeline (Init -> Setup -> Squish -> GenSecret
 * -> Query -> Answer -> Recover) and returns recovered bytes.
 *
 * Behaviour bifurcates on core::kPirCoreKernelsVendored: in stub
 * mode the kernel-check returns FAIL with the activation flag hint;
 * in vendored mode the full loop runs and recovered values match.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir.h"

namespace primihub::pir {
namespace {

// Builds an Options struct with the minimum fields OnExecute reads.
// Currently OnExecute does not consume any Options fields — it
// operates purely on the input PirDataType — so this is just the
// default-constructed shell.
Options MakeMinimalOptions() {
  Options o;
  o.code = "simple_pir_test";
  o.role = Role::CLIENT;
  return o;
}

// Convenience: build a db_content vector of decimal strings of size n,
// where entry i has value `(i * 13 + 7) % 256`. Deterministic so the
// expected recovered value is recomputed from the same formula.
std::vector<std::string> MakeDbContent(uint64_t n) {
  std::vector<std::string> v;
  v.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t val = (i * 13 + 7) % 256;
    v.push_back(std::to_string(val));
  }
  return v;
}

TEST(SimplePirOperatorTest, BifurcatesOnVendoring) {
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(64);
  input["query_indices"] = {"5", "10", "27"};
  PirDataType result;
  auto rc = op.OnExecute(input, &result);
  if (core::kPirCoreKernelsVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS);
    auto it = result.find("recovered");
    ASSERT_NE(it, result.end()) << "vendored mode must populate recovered";
    EXPECT_EQ(it->second.size(), 3u);
  } else {
    EXPECT_EQ(rc, retcode::FAIL)
        << "stub mode must surface the activation-flag error";
  }
}

TEST(SimplePirOperatorTest, FailsOnMissingDbContent) {
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, FailsOnMissingIndices) {
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(64);
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, FailsOnInvalidByteValue) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "stub mode bails out before reaching byte parse";
  }
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(64);
  input["db_content"][5] = "999";  // > 255, must FAIL
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, FailsOnNonDecimalIndex) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "stub mode bails out before reaching index parse";
  }
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(64);
  input["query_indices"] = {"abc"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, FailsOnIndexOutOfRange) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "stub mode bails out before reaching range check";
  }
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(64);
  input["query_indices"] = {"100"};  // N=64, idx 100 invalid
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, FailsOnRowAlignment) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "stub mode bails out before row-alignment check";
  }
  // N=100 -> l=10, not divisible by 8. Operator must FAIL with a
  // clear message about the row-alignment requirement.
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(100);
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(SimplePirOperatorTest, RetrievesByteValueCorrectly) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "end-to-end retrieval needs the kernel bridge";
  }
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  // N=64 -> l=8, m=8 (matches the protocol test's known-good config).
  const uint64_t n = 64;
  input["db_content"] = MakeDbContent(n);
  // Query a sampling of indices.
  input["query_indices"] = {"0", "1", "27", "63"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);

  auto it = result.find("recovered");
  ASSERT_NE(it, result.end());
  ASSERT_EQ(it->second.size(), 4u);

  // Expected values use the same formula as MakeDbContent.
  auto expected = [](uint64_t i) {
    return std::to_string((i * 13 + 7) % 256);
  };
  EXPECT_EQ(it->second[0], expected(0));
  EXPECT_EQ(it->second[1], expected(1));
  EXPECT_EQ(it->second[2], expected(27));
  EXPECT_EQ(it->second[3], expected(63));
}

TEST(SimplePirOperatorTest, BatchOfManyQueriesAllRetrieve) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs kernel bridge";
  }
  SimplePirOperator op(MakeMinimalOptions());
  PirDataType input;
  const uint64_t n = 64;
  input["db_content"] = MakeDbContent(n);
  // Query every single entry.
  std::vector<std::string> all_indices;
  for (uint64_t i = 0; i < n; ++i) all_indices.push_back(std::to_string(i));
  input["query_indices"] = all_indices;
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);

  auto it = result.find("recovered");
  ASSERT_NE(it, result.end());
  ASSERT_EQ(it->second.size(), n);
  for (uint64_t i = 0; i < n; ++i) {
    const std::string expected = std::to_string((i * 13 + 7) % 256);
    EXPECT_EQ(it->second[i], expected) << "mismatch at i=" << i;
  }
}

}  // namespace
}  // namespace primihub::pir
