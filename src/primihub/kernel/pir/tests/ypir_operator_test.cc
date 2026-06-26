/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * YpirOperator OnExecute integration test (task 7.3 chunk 14 / OnExecute
 * wiring). Drives a single-byte-element DB + query indices through the
 * PirDataType API and checks the operator runs the full ported YPIR SimplePIR
 * pipeline (offline hint + online AVX512-kernel answer + recursive packing +
 * Regev decrypt) and returns the correct base64-encoded recovered bytes. Also
 * pins the capability profile (is_real true, not a skeleton).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir.h"

#include <cstdint>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "base64.h"  // NOLINT

#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

Options MakeMinimalOptions() {
  Options o;
  o.code = "ypir_operator_test";
  o.role = Role::CLIENT;
  return o;
}

std::string B64Byte(std::uint8_t b) { return base64_encode(&b, 1); }

TEST(YpirOperatorTest, RoundtripSingleByteElements) {
  // 20 single-byte elements (all in row 0 at the poly_len=2048 db_cols=2048).
  const std::size_t m = 20;
  std::vector<std::string> db;
  db.reserve(m);
  for (std::size_t i = 0; i < m; ++i)
    db.push_back(B64Byte(static_cast<std::uint8_t>(i * 13u + 5u)));

  YpirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = db;
  // poly_len=2048 -> each query is ~seconds; keep the index set small.
  input["query_indices"] = {"5", "11"};

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 2u);
  EXPECT_EQ(result["recovered"][0], db[5]);
  EXPECT_EQ(result["recovered"][1], db[11]);
}

TEST(YpirOperatorTest, RejectsMultiByteElements) {
  YpirOperator op(MakeMinimalOptions());
  PirDataType input;
  std::vector<std::uint8_t> two = {1, 2};
  input["db_content"] = {base64_encode(two.data(), two.size())};
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(YpirOperatorTest, NotSkeletonAndReal) {
  EXPECT_FALSE(YpirOperator::kIsSkeleton);
  PirRegistry::EnsureRegistered();
  const PirCapabilities* caps =
      PirRegistry::Instance().GetCapabilities("ypir");
  ASSERT_NE(caps, nullptr);
  EXPECT_TRUE(caps->is_real);
}

}  // namespace
}  // namespace primihub::pir
