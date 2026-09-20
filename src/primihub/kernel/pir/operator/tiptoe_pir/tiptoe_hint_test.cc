/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Test for the LHE hint decompose/apply/recover pipeline (tiptoe chunk 1.1e).
 * Builds a synthetic SimplePIR hint + ternary secret, runs the full homomorphic
 * path (EncryptSecret -> DecomposeHint -> ApplyHint -> RecoverAS) and checks it
 * equals the cleartext limb recombination exactly. Real mode needs SEAL
 * (GFW-blocked on .50): manual + gated on --define=enable_tiptoe_real=1; without
 * it compiles to a single skipped test. Validated standalone vs SEAL 4.1 when
 * this chunk landed (5x4 hint, 5 limbs, exact match).
 */
#include "gtest/gtest.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED

#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_hint.h"

namespace primihub::pir::tiptoe {
namespace {

TEST(TiptoeHintTest, DecomposeApplyRecoverMatchesCleartext) {
  const std::uint64_t hint_rows = 5, cols = 4;
  const int elem_bits = 32;
  const std::uint64_t P = 65537, mask = 0xFFFFFFFFull;

  std::vector<std::uint64_t> H(hint_rows * cols);
  for (std::uint64_t r = 0; r < hint_rows; ++r)
    for (std::uint64_t c = 0; c < cols; ++c)
      H[r * cols + c] = (r * 2654435761u + c * 40503u + 12345u) & mask;
  const std::vector<std::uint64_t> s = {0, 1, 2, 1};

  Params params;
  std::vector<CipherBlob> enc_sk;
  const KeyBlob key = EncryptSecret(params, s, &enc_sk);
  ASSERT_EQ(enc_sk.size(), cols);

  const HintDecomp hd = DecomposeHint(params, H, hint_rows, cols, elem_bits);
  EXPECT_EQ(hd.pts.size(), static_cast<std::size_t>(LimbsFor(elem_bits)));

  const std::vector<std::vector<CipherBlob>> hint_cts =
      ApplyHint(params, hd, enc_sk);
  const std::vector<std::uint64_t> got =
      RecoverAS(params, key, hint_cts, hint_rows, elem_bits);
  ASSERT_EQ(got.size(), hint_rows);

  const int max_limbs = MaxLimbs(elem_bits), limbs = LimbsFor(elem_bits);
  for (std::uint64_t r = 0; r < hint_rows; ++r) {
    std::uint64_t exp = 0;
    for (int b = 0; b < limbs; ++b) {
      const int index = max_limbs - b - 1;
      std::uint64_t raw = 0;
      for (std::uint64_t c = 0; c < cols; ++c)
        raw += GetChunk(H[r * cols + c], index) * s[c];
      const std::uint64_t centered =
          static_cast<std::uint64_t>(FromModuloP<std::int64_t>(P, raw % P)) &
          mask;
      exp = (exp + centered * (std::uint64_t{1} << (4 * index))) & mask;
    }
    EXPECT_EQ(got[r] & mask, exp) << "row " << r;
  }
}

}  // namespace
}  // namespace primihub::pir::tiptoe

#else  // !PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoeHintTest, NeedsSeal) {
  GTEST_SKIP() << "build with --define=enable_tiptoe_real=1 + "
                  "--override_repository=underhood=<path> (needs SEAL toolchain)";
}

#endif
