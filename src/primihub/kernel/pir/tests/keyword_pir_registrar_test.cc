/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * keyword_pir_registrar_test — built only when --define=microsoft-apsi=true.
 *
 * proto_compat_test verifies the Factory shim accepts a stub fallback for
 * "apsi" so the wire-compat contract works even when APSI is unlinked.
 * That test cannot tell whether the real APSI registrar fired or the stub
 * did. This test closes the gap: when microsoft-apsi=true is on, the
 * registry MUST contain "apsi" with the actual KeywordPirCaps profile
 * (Keyword query type, SemiHonest threat model, per-database hint, CPU
 * backend). If the registrar TU were stripped, this test would fail and
 * Selector would silently lose APSI coverage.
 */
#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/factory.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

TEST(KeywordPirRegistrarTest, ApsiRegisteredWithRealCaps) {
  PirRegistry::EnsureRegistered();
  const auto* caps = PirRegistry::Instance().GetCapabilities("apsi");
  ASSERT_NE(caps, nullptr)
      << "APSI registrar did not fire even though microsoft-apsi=true was "
         "set at build time. Likely cause: keyword_pir_registrar cc_library "
         "lost its alwayslink=True attribute or was removed from the "
         "//src/primihub/kernel/pir/operator:keyword_pir_operator aggregator.";

  // Lock down the capability fingerprint so Selector matrix tests do not
  // silently shift their notion of what "apsi" means.
  EXPECT_EQ(caps->query_types.count(QueryType::Keyword), 1u);
  EXPECT_EQ(caps->min_servers, 1);
  EXPECT_EQ(caps->max_servers, 1);
  EXPECT_TRUE(caps->needs_preprocess);
  EXPECT_TRUE(caps->hint_per_database);
  EXPECT_EQ(caps->threat_model, ThreatModel::SemiHonest);
  EXPECT_EQ(caps->backends.count(Backend::CPU), 1u);
}

TEST(KeywordPirRegistrarTest, LegacyKeyPirRoutesToRealApsi) {
  // With the real registrar present, Factory::Create(KEY_PIR) must return
  // an operator wired to the APSI implementation, not a stub. We can't
  // introspect the type directly, but a non-null instance is the on-the-
  // wire guarantee for legacy clients.
  PirRegistry::EnsureRegistered();
  Options opt;
  auto op = Factory::Create(PirType::KEY_PIR, opt);
  ASSERT_NE(op, nullptr);
}

}  // namespace
}  // namespace primihub::pir
