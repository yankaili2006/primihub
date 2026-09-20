/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * PiranaPirOperator OnExecute end-to-end tests. Validates the
 * single-process self-contained retrieval path: caller passes a DB
 * (raw byte strings) + indices through PirDataType; operator runs the
 * full vendored PIRANA pipeline (Client gen_query → Server
 * gen_response → Client extract_answer) and returns raw byte strings.
 *
 * DB sizes are kept in the regime the upstream demo exercises
 * (hundreds-to-thousands of payloads): PirParms derives col_size /
 * pre_rotate from num_payloads vs poly degree 4096, and tiny
 * degenerate sizes (single digits) fall outside the parameter
 * selection asserts.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/pirana_pir/pirana_pir.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

Options MakeMinimalOptions() {
  Options o;
  o.code = "pirana_test";
  o.role = Role::CLIENT;
  return o;
}

// Deterministic DB: payload i is 8 bytes derived from i
// (0xii, ii+1, ...) so expected output is rederivable.
std::vector<std::string> MakeDb(std::size_t n, std::size_t payload_size) {
  std::vector<std::string> db;
  db.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::string payload(payload_size, '\0');
    for (std::size_t b = 0; b < payload_size; ++b) {
      payload[b] = static_cast<char>((i * 31u + b * 7u + 5u) & 0xFFu);
    }
    db.push_back(payload);
  }
  return db;
}

TEST(PiranaPirOperatorTest, RoundtripSingleQuery_512x8) {
  const std::size_t n = 512;
  const auto db = MakeDb(n, 8);

  PiranaPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"123"};

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 1u);
  EXPECT_EQ(result["recovered"][0], db[123])
      << "PIR retrieved wrong payload at index 123";
}

TEST(PiranaPirOperatorTest, RoundtripBatchQuery_1024x8) {
  const std::size_t n = 1024;
  const auto db = MakeDb(n, 8);

  PiranaPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = db;
  // Mix of start/middle/end indices.
  input["query_indices"] = {"0", "17", "512", "1023"};

  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  ASSERT_EQ(result["recovered"].size(), 4u);
  EXPECT_EQ(result["recovered"][0], db[0]);
  EXPECT_EQ(result["recovered"][1], db[17]);
  EXPECT_EQ(result["recovered"][2], db[512]);
  EXPECT_EQ(result["recovered"][3], db[1023]);
}

TEST(PiranaPirOperatorTest, IndexOutOfRange_Fails) {
  PiranaPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDb(512, 8);
  input["query_indices"] = {"99999"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(PiranaPirOperatorTest, MissingDbContent_Fails) {
  PiranaPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["query_indices"] = {"0"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST(PiranaPirOperatorTest, MissingQueryIndices_Fails) {
  PiranaPirOperator op(MakeMinimalOptions());
  PirDataType input;
  input["db_content"] = MakeDb(512, 8);
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

}  // namespace
}  // namespace primihub::pir
