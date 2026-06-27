/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Test for the LHE client encrypt-secret core (tiptoe chunk 1.1d). Real mode
 * needs SEAL (GFW-blocked on .50), so it is manual + gated on
 * --define=enable_tiptoe_real=1; without it compiles to a single skipped test.
 * Validated standalone against SEAL 4.1 when this chunk landed (8-element
 * ternary secret roundtrip PASS, ~16.5 KB/ct).
 */
#include "gtest/gtest.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED

#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_client.h"

namespace primihub::pir::tiptoe {
namespace {

TEST(TiptoeClientTest, EncryptSecretRoundtrip) {
  Params params;
  const std::vector<std::uint64_t> secret = {0, 1, 2, 1, 0, 2, 2, 1};

  std::vector<CipherBlob> cts;
  const KeyBlob key = EncryptSecret(params, secret, &cts);
  ASSERT_EQ(cts.size(), secret.size());
  EXPECT_FALSE(key.empty());

  for (std::size_t i = 0; i < secret.size(); ++i) {
    std::vector<std::uint64_t> coeffs;
    DecryptSquished(params, key, cts[i], &coeffs);
    ASSERT_EQ(coeffs.size(), params.N());
    EXPECT_EQ(coeffs[0], secret[i]) << "element " << i;
    EXPECT_EQ(coeffs[1], 0u) << "only coefficient 0 carries the secret (i=" << i
                             << ")";
  }
}

}  // namespace
}  // namespace primihub::pir::tiptoe

#else  // !PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoeClientTest, NeedsSeal) {
  GTEST_SKIP() << "build with --define=enable_tiptoe_real=1 + "
                  "--override_repository=underhood=<path> (needs SEAL toolchain)";
}

#endif
