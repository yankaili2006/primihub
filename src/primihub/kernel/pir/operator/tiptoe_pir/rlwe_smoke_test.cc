/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Smoke test for the vendored underhood/rlwe BFV-on-SimplePIR crypto core
 * (tiptoe chunk 1.1b). Exercises the homomorphic inner-product path Tiptoe
 * relies on: out = Enc(A)*w1 + Enc(B)*w2 (multiply_plain + add), serialize +
 * deserialize the result ciphertext, decrypt, and check the recovered
 * coefficient equals A*w1 + B*w2 mod p.
 *
 * Real mode needs the Microsoft SEAL toolchain (GFW-blocked on .50), so this
 * target is `manual` + gated on --define=enable_tiptoe_real=1:
 *   bazel test //src/primihub/kernel/pir/operator/tiptoe_pir:rlwe_smoke_test \
 *     --define=enable_tiptoe_real=1 \
 *     --override_repository=underhood=<path-to-ahenzinger/underhood>
 * Without the define it compiles to a single skipped test so analysis of the
 * default build is unaffected. The same logic was validated standalone against
 * SEAL 4.1 (got=25923=expect) when this chunk landed.
 */
#include "gtest/gtest.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED

#include <cstdint>
#include <vector>

#include "rlwe.h"  // @underhood//:rlwe (includes = ["rlwe"])

namespace {

void SetConst(plaintext_t* pt, context_t* ctx, std::uint64_t c0) {
  const std::size_t n = context_n(ctx);
  std::vector<std::uint64_t> v(n, 0);
  v[0] = c0;
  plaintext_set(pt, ctx, v.data(), n);
}

TEST(TiptoeRlweSmoke, InnerProductRoundtrip) {
  context_t* ctx = context_new();
  const std::size_t n = context_n(ctx);
  const std::uint64_t p = context_p(ctx);
  EXPECT_EQ(n, 2048u);
  EXPECT_EQ(p, 65537u);

  skey_t* key = key_new(ctx);
  const std::uint64_t A = 12345, B = 6789, w1v = 1, w2v = 2;

  plaintext_t *pa = plaintext_new(), *pb = plaintext_new();
  SetConst(pa, ctx, A);
  SetConst(pb, ctx, B);
  ciphertext_t *ca = ciphertext_new(), *cb = ciphertext_new();
  key_encrypt(key, pa, ca);
  key_encrypt(key, pb, cb);

  plaintext_t *w1 = plaintext_new(), *w2 = plaintext_new();
  SetConst(w1, ctx, w1v);
  SetConst(w2, ctx, w2v);

  ciphertext_t* out = ciphertext_new();
  ciphertext_t* cts[2] = {ca, cb};
  plaintext_t* pts[2] = {w1, w2};
  ciphertext_set_inner_product(ctx, out, cts, pts, 2);

  const std::size_t sz = ciphertext_size(out);
  std::vector<std::uint8_t> buf(sz);
  ciphertext_store(out, buf.data(), sz);
  ciphertext_t* out2 = ciphertext_new();
  ciphertext_load(ctx, out2, buf.data(), sz);

  plaintext_t* res = plaintext_new();
  key_decrypt(key, out2, res);
  std::vector<std::uint64_t> rv(n, 0);
  plaintext_dump(res, rv.data(), n);

  EXPECT_EQ(rv[0], (A * w1v + B * w2v) % p);

  plaintext_free(pa);
  plaintext_free(pb);
  plaintext_free(w1);
  plaintext_free(w2);
  plaintext_free(res);
  ciphertext_free(ca);
  ciphertext_free(cb);
  ciphertext_free(out);
  ciphertext_free(out2);
  key_free(key);
  context_free(ctx);
}

}  // namespace

#else  // !PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoeRlweSmoke, NeedsSeal) {
  GTEST_SKIP() << "build with --define=enable_tiptoe_real=1 + "
                  "--override_repository=underhood=<path> (needs SEAL toolchain)";
}

#endif
