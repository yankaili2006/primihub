/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/capabilities.h"

#include <gtest/gtest.h>
#include "src/primihub/kernel/pir/common.h"

namespace primihub::pir {
namespace {

PirCapabilities ValidCaps() {
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 1;
  c.max_servers = 1;
  c.backends = {Backend::CPU};
  return c;
}

TEST(PirCapabilitiesTest, ValidCapsPassCheck) {
  EXPECT_EQ(ValidCaps().Check(), "");
}

TEST(PirCapabilitiesTest, ZeroMinServersFails) {
  auto c = ValidCaps();
  c.min_servers = 0;
  EXPECT_NE(c.Check(), "");
}

TEST(PirCapabilitiesTest, MinGreaterThanMaxFails) {
  auto c = ValidCaps();
  c.min_servers = 3;
  c.max_servers = 2;
  EXPECT_NE(c.Check(), "");
}

TEST(PirCapabilitiesTest, EmptyQueryTypesFails) {
  auto c = ValidCaps();
  c.query_types.clear();
  EXPECT_NE(c.Check(), "");
}

TEST(PirCapabilitiesTest, HintWithoutPreprocessFails) {
  auto c = ValidCaps();
  c.needs_preprocess = false;
  c.typical_hint_size_bytes = 1000;
  EXPECT_NE(c.Check(), "");
}

TEST(PirCapabilitiesTest, EmptyBackendsFails) {
  auto c = ValidCaps();
  c.backends.clear();
  EXPECT_NE(c.Check(), "");
}

TEST(PirCapabilitiesTest, JsonContainsAllFields) {
  auto c = ValidCaps();
  c.recommended_max_db_size = 1000000;
  c.typical_query_comm_bytes = 14336;
  std::string j = c.ToJson();
  EXPECT_NE(j.find("\"query_types\""), std::string::npos);
  EXPECT_NE(j.find("\"min_servers\":1"), std::string::npos);
  EXPECT_NE(j.find("\"max_servers\":1"), std::string::npos);
  EXPECT_NE(j.find("\"recommended_max_db_size\":1000000"), std::string::npos);
  EXPECT_NE(j.find("\"typical_query_comm_bytes\":14336"), std::string::npos);
  EXPECT_NE(j.find("\"CPU\""), std::string::npos);
  EXPECT_NE(j.find("\"Index\""), std::string::npos);
}

TEST(PirCapabilitiesTest, DoublePirShapeIsValid) {
  // Smoke test: the shape we expect DoublePIR to register with.
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 2;
  c.max_servers = 2;
  c.needs_preprocess = true;
  c.hint_per_database = false;
  c.threat_model = ThreatModel::SemiHonestNonColluding;
  c.perf_class = PerfClass::Ms;
  c.recommended_max_db_size = 10000000000ULL;
  c.backends = {Backend::CPU, Backend::AVX2};
  c.typical_query_comm_bytes = 4096;
  c.typical_hint_size_bytes = 200000000ULL;
  EXPECT_EQ(c.Check(), "");
}

}  // namespace
}  // namespace primihub::pir
