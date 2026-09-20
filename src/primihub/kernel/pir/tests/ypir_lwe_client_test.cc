/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe_client_test — chunk 4d end-to-end. Wires key-gen + encrypt +
 * decrypt + scale_k decode and checks the LWE relation recovers the
 * message (noise |e| <= ceil(noise_width*4) ~= 111 << scale_k/2 = 2^23,
 * so rounding is exact). All RNGs seeded -> deterministic.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_client.h"

#include <array>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"

namespace primihub::pir::ypir {
namespace {

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  s[0] = b;
  return s;
}

// n is reduced for test speed; LWE correctness is independent of n.
LweParams SmallParams() {
  LweParams p = LweParams::Default();
  p.n = 8;
  return p;
}

// Round the noisy phase by scale_k and reduce mod pt_modulus.
std::uint32_t Decode(std::uint32_t phase, const LweParams& p) {
  const std::uint64_t k = p.ScaleK();
  return static_cast<std::uint32_t>(
      ((static_cast<std::uint64_t>(phase) + k / 2) / k) % p.pt_modulus);
}

TEST(YpirLweClientTest, KeyGen_Encrypt_Decrypt_RecoversMessage) {
  const auto params = SmallParams();
  auto entropy = ChaChaRng::FromSeed(Seed(1));
  const auto client = LweClient::New(params, entropy);
  ASSERT_EQ(client.sk().size(), params.n);

  for (std::uint32_t m : {0u, 1u, 42u, 200u, 255u}) {
    auto rng_pub = ChaChaRng::FromSeed(Seed(2));
    auto noise = ChaChaRng::FromSeed(Seed(3));
    const std::uint32_t pt = static_cast<std::uint32_t>(m * params.ScaleK());
    const auto ct = client.Encrypt(rng_pub, noise, pt);
    ASSERT_EQ(ct.size(), params.n + 1);
    EXPECT_EQ(Decode(client.Decrypt(ct), params), m) << "message " << m;
  }
}

TEST(YpirLweClientTest, EncryptMany_RecoversAllMessages) {
  const auto params = SmallParams();
  auto entropy = ChaChaRng::FromSeed(Seed(7));
  const auto client = LweClient::New(params, entropy);

  auto rng_pub = ChaChaRng::FromSeed(Seed(9));
  auto noise = ChaChaRng::FromSeed(Seed(11));
  const std::vector<std::uint32_t> msgs = {0, 17, 42, 99, 128, 200, 255, 1};
  std::vector<std::uint32_t> v_pt(params.n);
  for (std::size_t i = 0; i < params.n; ++i) {
    v_pt[i] = static_cast<std::uint32_t>(msgs[i] * params.ScaleK());
  }

  const auto ct = client.EncryptMany(rng_pub, noise, v_pt);
  ASSERT_EQ(ct.size(), params.n * params.n + params.n);

  for (std::size_t col = 0; col < params.n; ++col) {
    std::vector<std::uint32_t> col_ct(params.n + 1);
    for (std::size_t row = 0; row < params.n; ++row) {
      col_ct[row] = ct[row * params.n + col];
    }
    col_ct[params.n] = ct[params.n * params.n + col];
    EXPECT_EQ(Decode(client.Decrypt(col_ct), params), msgs[col]) << "col " << col;
  }
}

TEST(YpirLweClientTest, Deterministic_SameSeedSameKeyAndCiphertext) {
  const auto params = SmallParams();
  auto e1 = ChaChaRng::FromSeed(Seed(5));
  auto e2 = ChaChaRng::FromSeed(Seed(5));
  const auto c1 = LweClient::New(params, e1);
  const auto c2 = LweClient::New(params, e2);
  EXPECT_EQ(c1.sk(), c2.sk());

  auto p1 = ChaChaRng::FromSeed(Seed(6));
  auto n1 = ChaChaRng::FromSeed(Seed(8));
  auto p2 = ChaChaRng::FromSeed(Seed(6));
  auto n2 = ChaChaRng::FromSeed(Seed(8));
  const std::uint32_t pt = static_cast<std::uint32_t>(42u * params.ScaleK());
  EXPECT_EQ(c1.Encrypt(p1, n1, pt), c2.Encrypt(p2, n2, pt));
}

TEST(YpirLweClientTest, FromSecretKey_DecryptHandConstructed) {
  const auto params = SmallParams();
  const std::vector<std::uint32_t> sk = {3, 5, 9, 2, 1, 4, 6, 8};
  const auto client = LweClient::FromSecretKey(params, sk);
  const std::vector<std::uint32_t> a = {7, 11, 4, 6, 2, 3, 5, 1};
  std::uint32_t sum = 0;
  for (std::size_t i = 0; i < 8; ++i) sum += a[i] * sk[i];
  const std::uint32_t phase = 12345;
  std::vector<std::uint32_t> ct = a;
  ct.push_back((0u - sum) + phase);  // b makes the phase exactly `phase`
  EXPECT_EQ(client.Decrypt(ct), phase);
}

}  // namespace
}  // namespace primihub::pir::ypir
