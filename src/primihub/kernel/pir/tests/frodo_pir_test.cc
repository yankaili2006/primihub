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

#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_pir.h"
#include "src/primihub/kernel/pir/operator/registry.h"

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


// ---- Chunk 8: registry-driven tests --------------------------

class FrodoPirRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override { PirRegistry::EnsureRegistered(); }
};

TEST_F(FrodoPirRegistryTest, RegisteredUnderFrodoPirName) {
  const auto algos = PirRegistry::Instance().ListAlgorithms();
  bool found = false;
  for (const auto& a : algos) {
    if (a == "frodo_pir") { found = true; break; }
  }
  EXPECT_TRUE(found) << "frodo_pir missing from PirRegistry";
}

TEST_F(FrodoPirRegistryTest, CapabilityProfileMatchesPaper) {
  const auto* caps = PirRegistry::Instance().GetCapabilities("frodo_pir");
  ASSERT_NE(caps, nullptr);
  // From de Castro & Lee, PETS 2023.
  EXPECT_TRUE(caps->is_real) << "caps.is_real must be true post chunk 7";
  EXPECT_EQ(caps->query_types.count(QueryType::Index), 1u);
  EXPECT_EQ(caps->min_servers, 1u);
  EXPECT_EQ(caps->max_servers, 1u);
  EXPECT_TRUE(caps->needs_preprocess);
  EXPECT_TRUE(caps->hint_per_database);
  EXPECT_EQ(caps->threat_model, ThreatModel::SemiHonest);
  EXPECT_EQ(caps->perf_class, PerfClass::Ms);
  EXPECT_GE(caps->recommended_max_db_size, 100000000ULL);
  EXPECT_GT(caps->typical_query_comm_bytes, 0u);
  EXPECT_GT(caps->typical_hint_size_bytes, 0u);
  EXPECT_EQ(caps->Check(), "");
}

TEST_F(FrodoPirRegistryTest, FactoryReturnsNonNull) {
  Options opt = MakeMinimalOptions();
  auto op = PirRegistry::Instance().Create("frodo_pir", opt);
  ASSERT_NE(op, nullptr);
}

TEST_F(FrodoPirRegistryTest, NotSkeletonAfterChunk7) {
  EXPECT_FALSE(FrodoPirOperator::kIsSkeleton)
      << "chunk 7 must flip kIsSkeleton=false";
}

TEST_F(FrodoPirRegistryTest, RegistryDrivenEndToEndRetrieval) {
  Options opt = MakeMinimalOptions();
  auto op = PirRegistry::Instance().Create("frodo_pir", opt);
  ASSERT_NE(op, nullptr);
  const std::vector<std::uint8_t> orig = {0x11u, 0x22u, 0x33u, 0x44u,
                                          0x55u, 0x66u, 0x77u, 0x88u};
  std::vector<std::string> db;
  for (auto b : orig) db.push_back(B64({b}));
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"0", "3", "7"};
  PirDataType result;
  ASSERT_EQ(op->OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 3u);
  EXPECT_EQ(result["recovered"][0], db[0]);
  EXPECT_EQ(result["recovered"][1], db[3]);
  EXPECT_EQ(result["recovered"][2], db[7]);
}
}  // namespace
}  // namespace primihub::pir
