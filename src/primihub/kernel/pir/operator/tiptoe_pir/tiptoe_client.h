/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * LHE client core ported from underhood/underhood/client.go + the encryptSecret
 * method of secret.go (tiptoe chunk 1.1d). This is the rlwe-only part: encrypt
 * the inner (SimplePIR) ternary secret vector into per-element squished LHE
 * ciphertexts -- the "hint query" the client sends so the server can compute
 * H*s homomorphically. The SimplePIR query/recover orchestration (HintQuery /
 * Query / Recover over pir.Client) integrates with primihub's ported SimplePIR
 * core in chunk 1.1f.
 *
 * Real mode only: pulls in @underhood//:rlwe via tiptoe_params.h (needs SEAL),
 * so this header is compiled only under --define=enable_tiptoe_real=1.
 * See docs/pir/tiptoe-port-plan.md.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_CLIENT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_CLIENT_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_params.h"
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_secret.h"

namespace primihub::pir::tiptoe {

// Serialized outer LHE secret key / ciphertext (underhood KeyBlob / CipherBlob).
using KeyBlob = std::vector<std::uint8_t>;
using CipherBlob = std::vector<std::uint8_t>;

// Encrypt the inner SimplePIR secret (a column vector of ternary entries in
// {0,1,2}) under a fresh outer LHE key, one squished ciphertext per element
// (underhood secret.go encryptSecret). The element value is placed in
// coefficient 0 of an N-slot BFV plaintext; the rest are zero. Returns the
// serialized outer key (Store) and fills *cts with the per-element squished
// ciphertexts (EncryptSquishedSlice). The key blob is kept by the caller to
// later decrypt H*s (chunk 1.1e recoverAS).
inline KeyBlob EncryptSecret(const Params& params,
                             const std::vector<std::uint64_t>& inner_secret,
                             std::vector<CipherBlob>* cts) {
  assert(cts != nullptr);
  assert(params.P() >= kSecretMax);  // "P is too small to encode secret"

  skey_t* outer = key_new(params.ctx());
  const std::size_t n = params.N();
  std::vector<std::uint64_t> vals(n, 0);

  cts->clear();
  cts->reserve(inner_secret.size());
  for (const std::uint64_t e : inner_secret) {
    assert(InRange<std::uint64_t>(e));  // secret must be ternary
    plaintext_t* pt = plaintext_new();
    vals[0] = e;  // coefficients 1..n-1 stay zero
    plaintext_set(pt, params.ctx(), vals.data(), n);

    const std::size_t sz = key_encrypt_squished_size(outer, pt);
    CipherBlob blob(sz);
    key_encrypt_squished(outer, pt, blob.data(), sz);
    cts->push_back(std::move(blob));
    plaintext_free(pt);
  }

  const std::size_t ksz = key_size(outer);
  KeyBlob key_blob(ksz);
  key_store(outer, key_blob.data(), ksz);
  key_free(outer);
  return key_blob;
}

// Decrypt a squished LHE ciphertext under the stored outer key, returning its
// plaintext coefficients (length params.N()). Mirrors the load(key) + load(ct)
// + decrypt path; reused by the hint recover (chunk 1.1e) and by tests.
inline void DecryptSquished(const Params& params, const KeyBlob& key_blob,
                            const CipherBlob& ct_blob,
                            std::vector<std::uint64_t>* coeffs_out) {
  assert(coeffs_out != nullptr);
  skey_t* key = key_new(params.ctx());
  key_load(params.ctx(), key,
           const_cast<std::uint8_t*>(key_blob.data()), key_blob.size());

  ciphertext_t* ct = ciphertext_new();
  ciphertext_load(params.ctx(), ct,
                  const_cast<std::uint8_t*>(ct_blob.data()), ct_blob.size());

  plaintext_t* pt = plaintext_new();
  key_decrypt(key, ct, pt);

  const std::size_t n = params.N();
  coeffs_out->assign(n, 0);
  plaintext_dump(pt, coeffs_out->data(), n);

  plaintext_free(pt);
  ciphertext_free(ct);
  key_free(key);
}

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_CLIENT_H_
