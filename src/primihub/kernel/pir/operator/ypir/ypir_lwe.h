/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe — RNG-free LWE encryption/decryption cores from upstream
 * menonsamir/ypir@a73e550a src/lwe.rs. Chunk 4c of the YPIR port
 * (see docs/pir/ypir-port-plan.md).
 *
 * Upstream's LWEClient holds (lwe_params, sk) and exposes
 * encrypt / encrypt_many / decrypt. The arithmetic is pure u32-wrapping
 * (== mod 2^32, the default LWE modulus) dot-products; the only
 * randomness is the public sample vector `a` (drawn from a seeded
 * ChaCha20Rng) and the noise `e` (drawn from DiscreteGaussian). We port
 * the cores with `a`/`e` injected as parameters, keeping them pure and
 * roundtrip-testable.
 *
 * Deferred (RNG-coupled): LWEClient::new (secret-key generation) needs a
 * rand_chacha::ChaCha20Rng-byte-compatible PRNG, which is not yet ported
 * (frodo chunk 2b-iii). DiscreteGaussian (chunk 4b) and
 * NegacyclicMatrixU32 (ypir_negacyclic) are already available; once the
 * compatible ChaCha lands, an LWEClient wrapper can draw `a`/`e` and call
 * these cores.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_H_

#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Single-message LWE encryption (upstream LWEClient::encrypt core).
// `a` is the n public uniform samples; `e` is the noise. n = sk.size()
// = a.size(). Returns the (n+1)-length ciphertext [a_0..a_{n-1}, b] with
// b = (-<a, sk>) + pt + e, all u32-wrapping.
std::vector<std::uint32_t> LweEncrypt(const std::vector<std::uint32_t>& sk,
                                      const std::vector<std::uint32_t>& a,
                                      std::uint32_t pt, std::uint32_t e);

// Batched LWE encryption (upstream LWEClient::encrypt_many core). `a`,
// `v_pt`, `e` each have length n = sk.size(). Builds the negacyclic
// expansion nega_a (n*n, via NegacyclicMatrixU32) and a last row where
// last_row[col] = (-<nega_a[:,col], sk>) + v_pt[col] + e[col]. Returns
// the (n*n + n)-length ciphertext [nega_a (row-major), last_row].
std::vector<std::uint32_t> LweEncryptMany(
    const std::vector<std::uint32_t>& sk,
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& v_pt,
    const std::vector<std::uint32_t>& e);

// LWE decryption (upstream LWEClient::decrypt). Returns the noisy phase
// <ct[0..n], sk> + ct[n] (u32-wrapping); the caller rounds by scale_k to
// recover the plaintext. Requires ct.size() >= sk.size() + 1.
std::uint32_t LweDecrypt(const std::vector<std::uint32_t>& sk,
                         const std::vector<std::uint32_t>& ct);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_H_
