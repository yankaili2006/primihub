/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * FrodoPirOperator OnExecute integration tests (chunk 7). Validates
 * the single-process self-contained retrieval path: caller passes
 * a DB + indices through PirDataType; operator runs the full chunk-5
 * api layer (Shard + QueryParams + Query + Response) and returns
 * base64-encoded recovered values.
 *
 * Mirrors the SimplePirOperator test pattern; FrodoPIR has no
 * "vendored kernel" activation flag — every chunk-1..5 helper is
 * compile-time and always present.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "base64.h"  // NOLINT

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_pir.h"

namespace primihub::pir {
namespace {

Options MakeMinimalOptions() {
  Options o;
  o.code = "frodo_pir_test";
  o.role = Role::CLIENT;
  return o;
}

std::string B64(const std::vector<std::uint8_t>& bytes) {
  std::string s(bytes.begin(), bytes.end());
  return base64_encode(
      reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

// Builds m base64-encoded 1-byte DB entries with deterministic
// bytes (i * 17 + 13) mod 256 so the expected output is rederivable
// from the same formula.
std::vector<std::string> MakeDbContent(std::size_t m) {
  std::vector<std::string> v;
  v.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    const std::uint8_t b =
        static_cast<std::uint8_t>((i * 17u + 13u) & 0xFFu);
    v.push_back(B64({b}));
  }
  return v;
}

// ---- Chunk 7 OnExecute path ------------------------------------

TEST(FrodoPirOperatorTest, RoundtripSingleIndex_4Element) {
  const std::vector<std::uint8_t> orig = {0xAAu, 0xBBu, 0xCCu, 0xDDu};
  std::vector<std::string> db;
  for (auto b : orig) db.push_back(B64({b}));

  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"2"};  // expect 0xCC

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 1u);
  EXPECT_EQ(result["recovered"][0], db[2])
      << "operator-level PIR retrieved wrong byte at index 2; got "
      << "base64 '" << result["recovered"][0]
      << "', expected '" << db[2] << "'";
}

TEST(FrodoPirOperatorTest, RoundtripMultiIndex_16Element) {
  const std::size_t m = 16;
  const auto db = MakeDbContent(m);

  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = db;
  // Query a mix of start/middle/end indices.
  input["query_indices"] = {"0", "5", "10", "15"};

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 4u);
  EXPECT_EQ(result["recovered"][0], db[0]);
  EXPECT_EQ(result["recovered"][1], db[5]);
  EXPECT_EQ(result["recovered"][2], db[10]);
  EXPECT_EQ(result["recovered"][3], db[15]);
}

TEST(FrodoPirOperatorTest, MissingDbContent_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(FrodoPirOperatorTest, MissingQueryIndices_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(4);
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(FrodoPirOperatorTest, MixedElementSizes_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  // First entry decodes to 1 byte, second to 2 bytes — must reject.
  input["db_content"] = {B64({0x01u}), B64({0x01u, 0x02u})};
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(FrodoPirOperatorTest, IndexOutOfRange_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(4);
  input["query_indices"] = {"99"};  // m = 4
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(FrodoPirOperatorTest, GarbageIndex_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(4);
  input["query_indices"] = {"not-a-number"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(FrodoPirOperatorTest, NullResult_Fails) {
  FrodoPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDbContent(4);
  input["query_indices"] = {"0"};
  EXPECT_EQ(op.OnExecute(input, nullptr), retcode::FAIL);
}

}  // namespace
}  // namespace primihub::pir
