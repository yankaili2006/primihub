/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/registry.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"

namespace primihub::pir {
namespace {

class FakeOperator : public BasePirOperator {
 public:
  explicit FakeOperator(const Options& options) : BasePirOperator(options) {}
  retcode OnExecute(const PirDataType&, PirDataType*) override {
    return retcode::SUCCESS;
  }
};

PirCapabilities BasicCaps() {
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 1;
  c.max_servers = 1;
  c.backends = {Backend::CPU};
  return c;
}

TEST(PirRegistryTest, RegisterAndCreate) {
  auto ok = PirRegistry::Instance().Register(
      "fake_test_alpha",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      BasicCaps());
  EXPECT_TRUE(ok);

  Options opt;
  auto op = PirRegistry::Instance().Create("fake_test_alpha", opt);
  EXPECT_NE(op, nullptr);
}

TEST(PirRegistryTest, CreateUnknownReturnsNull) {
  Options opt;
  auto op = PirRegistry::Instance().Create("no_such_algo_xyz", opt);
  EXPECT_EQ(op, nullptr);
}

TEST(PirRegistryTest, GetCapabilities) {
  PirRegistry::Instance().Register(
      "fake_test_beta",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      BasicCaps());

  const auto* caps = PirRegistry::Instance().GetCapabilities("fake_test_beta");
  ASSERT_NE(caps, nullptr);
  EXPECT_EQ(caps->min_servers, 1u);
  EXPECT_EQ(caps->max_servers, 1u);
  EXPECT_TRUE(caps->query_types.count(QueryType::Index));

  EXPECT_EQ(PirRegistry::Instance().GetCapabilities("no_such_algo_xyz"),
            nullptr);
}

TEST(PirRegistryTest, RegisterDuplicateRejected) {
  auto ok1 = PirRegistry::Instance().Register(
      "fake_test_dup",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      BasicCaps());
  EXPECT_TRUE(ok1);
  auto ok2 = PirRegistry::Instance().Register(
      "fake_test_dup",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      BasicCaps());
  EXPECT_FALSE(ok2);  // duplicate registration must be refused
}

TEST(PirRegistryTest, ListAlgorithmsContainsRegistered) {
  PirRegistry::Instance().Register(
      "fake_test_list",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      BasicCaps());
  auto names = PirRegistry::Instance().ListAlgorithms();
  bool found = false;
  for (const auto& n : names) {
    if (n == "fake_test_list") {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);
}

TEST(PirRegistryTest, RejectInconsistentCapabilities) {
  PirCapabilities bad = BasicCaps();
  bad.min_servers = 5;
  bad.max_servers = 2;
  auto ok = PirRegistry::Instance().Register(
      "fake_test_bad",
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new FakeOperator(o));
      },
      bad);
  EXPECT_FALSE(ok);
}

TEST(PirRegistryTest, EnsureRegisteredCallable) {
  EXPECT_NO_THROW(PirRegistry::EnsureRegistered());
}

}  // namespace
}  // namespace primihub::pir
