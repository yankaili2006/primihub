/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_client chunk 12b-3: YClient::GenerateQueryImpl +
 * DecodeResponse. Both are exercised at the client level (the full cross-server
 * E2E awaits the server online path, 10d-10g):
 *   - GenerateQueryImpl(packing): the query ciphertext's noise is e*poly_len^-1,
 *     so decrypting and multiplying by poly_len (= factor^-1) cancels the
 *     scaling, leaving Delta*one_hot + e; rescaling recovers the one-hot
 *     selector exactly.
 *   - DecodeResponse: a synthetic LWE answer encrypting known values under the
 *     client's sk_reg is decoded back to those values.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params P8() {
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, 1, 1, 1, 0,
                      0);
}

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirYClientTest, GenerateQueryImplPackingRecoversOneHot) {
  NttContext ctx(P8());
  const Params& p = ctx.params();
  ChaChaRng key = ChaChaRng::FromSeed(Seed(1));
  const Client client(ctx, /*hamming=*/2, key);
  const YClient yc(ctx, client);

  const std::size_t dim_log2 = 1;     // 2 blocks
  const std::size_t index = 11;       // block 1, coeff 3 (poly_len=8)
  ChaChaRng noise = ChaChaRng::FromSeed(Seed(7));
  const std::vector<PolyMatrixRaw> q =
      yc.GenerateQueryImpl(static_cast<std::uint8_t>(1), dim_log2, /*packing=*/true, index, noise);
  ASSERT_EQ(q.size(), static_cast<std::size_t>(1) << dim_log2);

  const std::uint64_t poly_len = p.poly_len;
  for (std::size_t blk = 0; blk < q.size(); ++blk) {
    const PolyMatrixRaw dec =
        ctx.FromNtt(client.DecryptMatrixReg(ctx.ToNtt(q[blk])));
    for (std::size_t z = 0; z < poly_len; ++z) {
      // dec = factor*(Delta*one_hot + e); *poly_len cancels factor.
      const std::uint64_t v = static_cast<std::uint64_t>(
          (static_cast<__uint128_t>(dec.data[z]) * poly_len) % p.modulus);
      const std::uint64_t rec = Rescale(v, p.modulus, p.pt_modulus);
      const std::uint64_t expect =
          (blk == index / poly_len && z == index % poly_len) ? 1u : 0u;
      EXPECT_EQ(rec, expect) << "blk=" << blk << " z=" << z;
    }
  }
}

TEST(YpirYClientTest, DecodeResponseRecoversLweValues) {
  NttContext ctx(P8());
  const Params& p = ctx.params();
  ChaChaRng key = ChaChaRng::FromSeed(Seed(1));
  const Client client(ctx, 2, key);
  const YClient yc(ctx, client);

  const std::uint64_t* sk = client.SkReg().data.data();
  const std::size_t db_cols = 3;
  const std::uint64_t delta = p.modulus / p.pt_modulus;
  const std::vector<std::uint64_t> vals = {5, 100, 200};  // < pt_modulus

  // response is (poly_len+1) x db_cols: pick a uniformly, set b so the phase
  // <a,sk> + b == Delta*val (mod q).
  std::vector<std::uint64_t> resp((p.poly_len + 1) * db_cols, 0);
  ChaChaRng rng = ChaChaRng::FromSeed(Seed(9));
  for (std::size_t col = 0; col < db_cols; ++col) {
    __uint128_t dot = 0;
    for (std::size_t i = 0; i < p.poly_len; ++i) {
      const std::uint64_t a =
          static_cast<std::uint64_t>(rng.NextU32()) % p.modulus;
      resp[i * db_cols + col] = a;
      dot += static_cast<__uint128_t>(a) * sk[i];
    }
    const std::uint64_t dot_mod = static_cast<std::uint64_t>(dot % p.modulus);
    const std::uint64_t target =
        static_cast<std::uint64_t>((static_cast<__uint128_t>(delta) * vals[col]) %
                                   p.modulus);
    resp[p.poly_len * db_cols + col] =
        (target + p.modulus - dot_mod) % p.modulus;
  }

  const std::vector<std::uint64_t> out = yc.DecodeResponse(resp, db_cols);
  ASSERT_EQ(out.size(), db_cols);
  for (std::size_t col = 0; col < db_cols; ++col)
    EXPECT_EQ(out[col], vals[col]) << "col=" << col;
}

}  // namespace
}  // namespace primihub::pir::ypir
