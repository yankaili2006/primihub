/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe_client — the LWEClient assembly from upstream
 * menonsamir/ypir@a73e550a src/lwe.rs, wiring together the now-ported
 * pieces: LweParams (chunk 3a), the RNG-free LWE cores (chunk 4c),
 * DiscreteGaussian (chunk 4b) and the rand_chacha-compatible ChaChaRng
 * (chunk 2b-iii) + NegacyclicMatrixU32 (chunk 6a). Chunk 4d.
 *
 * Upstream draws randomness from ChaCha20Rng::from_entropy internally;
 * here the RNGs are injected (a ChaChaRng for the public sample vector
 * `a`, a ChaChaRng for the Gaussian noise) so the whole client is
 * reproducible and roundtrip-testable. Real callers seed the noise RNG
 * from entropy and the public RNG from the agreed public seed.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_CLIENT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_CLIENT_H_

#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"

namespace primihub::pir::ypir {

class LweClient {
 public:
  // Generate a fresh secret key: sk[i] = DiscreteGaussian(noise_width)
  // .Sample(modulus, entropy.NextU64()) for i in [0, params.n), matching
  // LWEClient::new (which draws `entropy` from ChaCha20Rng::from_entropy).
  static LweClient New(const LweParams& params, ChaChaRng& entropy);

  // Construct from an explicit secret key (n = sk.size()).
  static LweClient FromSecretKey(LweParams params,
                                 std::vector<std::uint32_t> sk);

  const LweParams& params() const { return params_; }
  const std::vector<std::uint32_t>& sk() const { return sk_; }

  // Single-message encrypt (LWEClient::encrypt): noise e drawn first
  // from `noise`, then the n public samples a[i] = rng_pub.NextU32().
  // Returns the (n+1) ciphertext.
  std::vector<std::uint32_t> Encrypt(ChaChaRng& rng_pub, ChaChaRng& noise,
                                     std::uint32_t pt) const;

  // Batched encrypt (LWEClient::encrypt_many): n public samples a drawn
  // first from `rng_pub`, then n noise values from `noise` (column
  // order). Returns the (n*n + n) ciphertext. v_pt.size() must == n.
  std::vector<std::uint32_t> EncryptMany(
      ChaChaRng& rng_pub, ChaChaRng& noise,
      const std::vector<std::uint32_t>& v_pt) const;

  // Decrypt (LWEClient::decrypt): the noisy phase <ct[:n], sk> + ct[n].
  std::uint32_t Decrypt(const std::vector<std::uint32_t>& ct) const;

 private:
  LweParams params_{};
  std::vector<std::uint32_t> sk_;
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_CLIENT_H_
