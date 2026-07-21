/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_client.h"

#include <cstddef>
#include <utility>

#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe.h"

namespace primihub::pir::ypir {

LweClient LweClient::New(const LweParams& params, ChaChaRng& entropy) {
  const auto dg = DiscreteGaussian::Init(params.noise_width);
  std::vector<std::uint32_t> sk;
  sk.reserve(params.n);
  for (std::size_t i = 0; i < params.n; ++i) {
    sk.push_back(static_cast<std::uint32_t>(
        dg.Sample(params.modulus, entropy.NextU64())));
  }
  return FromSecretKey(params, std::move(sk));
}

LweClient LweClient::FromSecretKey(LweParams params,
                                   std::vector<std::uint32_t> sk) {
  LweClient c;
  c.params_ = params;
  c.sk_ = std::move(sk);
  return c;
}

std::vector<std::uint32_t> LweClient::Encrypt(ChaChaRng& rng_pub,
                                              ChaChaRng& noise,
                                              std::uint32_t pt) const {
  const auto dg = DiscreteGaussian::Init(params_.noise_width);
  // Upstream draws the noise first, then the public samples.
  const std::uint32_t e =
      static_cast<std::uint32_t>(dg.Sample(params_.modulus, noise.NextU64()));
  std::vector<std::uint32_t> a(params_.n);
  for (std::size_t i = 0; i < params_.n; ++i) a[i] = rng_pub.NextU32();
  return LweEncrypt(sk_, a, pt, e);
}

std::vector<std::uint32_t> LweClient::EncryptMany(
    ChaChaRng& rng_pub, ChaChaRng& noise,
    const std::vector<std::uint32_t>& v_pt) const {
  const auto dg = DiscreteGaussian::Init(params_.noise_width);
  // Upstream draws all public samples first, then per-column noise.
  std::vector<std::uint32_t> a(params_.n);
  for (std::size_t i = 0; i < params_.n; ++i) a[i] = rng_pub.NextU32();
  std::vector<std::uint32_t> e(params_.n);
  for (std::size_t i = 0; i < params_.n; ++i) {
    e[i] = static_cast<std::uint32_t>(dg.Sample(params_.modulus, noise.NextU64()));
  }
  return LweEncryptMany(sk_, a, v_pt, e);
}

std::uint32_t LweClient::Decrypt(const std::vector<std::uint32_t>& ct) const {
  return LweDecrypt(sk_, ct);
}

}  // namespace primihub::pir::ypir
