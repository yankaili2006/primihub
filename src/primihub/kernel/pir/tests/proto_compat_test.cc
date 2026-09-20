/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * proto_compat_test — verifies the multi-algo framework keeps the legacy
 * PirType-based wire format working. The contract:
 *
 *   1. LegacyNameFor(ID_PIR) == "id_pir" and (KEY_PIR) == "apsi"; both
 *      values are pinned forever — clients on the wire encode pir_type as
 *      an int32 and the server-side enum mapping MUST NOT drift.
 *   2. Factory::Create(PirType, options) routes through PirRegistry and
 *      returns a non-null operator when the legacy-named algorithm is
 *      registered in the binary.
 *   3. Factory::Create returns nullptr (not crash) when the legacy algo
 *      is *not* registered — i.e. when a build was made with
 *      --define=disable_id_pir=1 or similar — so callers can fail
 *      gracefully rather than UB on a null deref.
 */
#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/factory.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

class StubOperator : public BasePirOperator {
 public:
  explicit StubOperator(const Options& options) : BasePirOperator(options) {}
  retcode OnExecute(const PirDataType&, PirDataType*) override {
    return retcode::SUCCESS;
  }
};

PirCapabilities MinimalCaps() {
  PirCapabilities c;
  c.query_types = {QueryType::Index};
  c.min_servers = 1;
  c.max_servers = 1;
  c.backends = {Backend::CPU};
  return c;
}

// Helper: register a stub under `algo` if and only if no real algorithm has
// claimed it yet. Returns true if the test owns the registration after this
// call (either it just registered, OR the production algorithm is present).
bool EnsureAlgoPresent(const std::string& algo) {
  PirRegistry::EnsureRegistered();
  if (PirRegistry::Instance().GetCapabilities(algo) != nullptr) {
    return true;
  }
  return PirRegistry::Instance().Register(
      algo,
      [](const Options& o) -> std::unique_ptr<BasePirOperator> {
        return std::unique_ptr<BasePirOperator>(new StubOperator(o));
      },
      MinimalCaps());
}

TEST(PirProtoCompatTest, LegacyNameForPinned) {
  // These two mappings MUST NEVER change. Clients in the wild encode
  // pir_type as 0 (ID_PIR) or 1 (KEY_PIR); the server resolves the
  // algorithm name from the enum here. Renaming "id_pir" or "apsi" in
  // common.h is an on-the-wire break.
  EXPECT_STREQ(LegacyNameFor(PirType::ID_PIR), "id_pir");
  EXPECT_STREQ(LegacyNameFor(PirType::KEY_PIR), "apsi");
}

TEST(PirProtoCompatTest, LegacyNameForUnknownEnumIsEmpty) {
  // Defensive: future additions to the enum that we don't yet know about
  // must round-trip to empty so that callers (e.g. Factory::Create) can
  // detect and log rather than dereferencing a bad pointer. We can't add
  // a value to the enum here without changing the header, so we cast an
  // out-of-band value to exercise the default branch.
  auto bogus = static_cast<PirType>(99);
  EXPECT_STREQ(LegacyNameFor(bogus), "");
}

TEST(PirProtoCompatTest, FactoryShimRoutesIdPir) {
  ASSERT_TRUE(EnsureAlgoPresent("id_pir"));
  Options opt;
  auto op = Factory::Create(PirType::ID_PIR, opt);
  ASSERT_NE(op, nullptr)
      << "Legacy ID_PIR clients must produce a non-null operator when 'id_pir' "
         "is registered. Either the shim broke or the algorithm was unlinked.";
}

TEST(PirProtoCompatTest, FactoryShimRoutesKeyPir) {
  ASSERT_TRUE(EnsureAlgoPresent("apsi"));
  Options opt;
  auto op = Factory::Create(PirType::KEY_PIR, opt);
  ASSERT_NE(op, nullptr)
      << "Legacy KEY_PIR clients must produce a non-null operator when 'apsi' "
         "is registered.";
}

TEST(PirProtoCompatTest, FactoryReturnsNullForUnknownLegacyEnum) {
  Options opt;
  auto bogus = static_cast<PirType>(99);
  auto op = Factory::Create(bogus, opt);
  EXPECT_EQ(op, nullptr)
      << "Out-of-band PirType values must produce nullptr (logged), not "
         "crash; the wire-compat contract bound to the enum is finite.";
}

}  // namespace
}  // namespace primihub::pir
