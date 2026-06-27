/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Self-contained LHE-on-SimplePIR retrieval (tiptoe chunk 1.1f) -- the v1 core
 * Tiptoe operator. Combines a ternary-secret SimplePIR linear layer with the
 * LHE hint machinery (chunks 1.1c-1.1e):
 *
 *   offline:  H = D*A ;  client sends Enc(s) ;  server computes Enc(H*s)
 *             (ApplyHint) ;  client recovers interm = H*s (RecoverAS, top limbs)
 *   online:   query col j: qu = A*s + Delta*u_j ;  server ans = D*qu ;
 *             client D[i][j] = round((ans[i] - interm[i]) / Delta) mod p
 *
 * The low bits of H*s dropped by the limb truncation stay below Delta/2, so
 * recovery is exact. SimplePIR p=256 (one byte/entry), q=2^32, Delta=2^24,
 * elem_bits=32 (top 5 of 8 limbs). The ternary secret and the per-query LWE
 * error are sampled from a CSPRNG (OpenSSL RAND_bytes), so the query is IND-CPA
 * and the retrieved index is hidden (chunk 1.1g). The public matrix A is still
 * expanded from a seed via splitmix64 -- fine for a public LWE matrix, though a
 * CSPRNG-expanded public seed would be stronger. Remaining follow-up: reuse
 * primihub's SimplePIR core::Database for the linear layer (needs extending it
 * to ternary secrets). Real mode only (LHE needs SEAL). See
 * docs/pir/tiptoe-port-plan.md.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LHE_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LHE_PIR_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <openssl/rand.h>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_client.h"
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_hint.h"
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_params.h"

namespace primihub::pir::tiptoe {

// SimplePIR plaintext modulus: one byte per DB entry.
inline constexpr std::uint64_t kSimplePirP = 256;
// q = 2^32 (Z_q is uint32, arithmetic done mod 2^32). Delta = q / p = 2^24.
inline constexpr std::uint64_t kSimplePirDelta = (std::uint64_t{1} << 32) / kSimplePirP;
inline constexpr int kSimplePirElemBits = 32;

// Bound on each LWE query-error coordinate (chunk 1.1g). Recovery stays exact
// while |D.e| < Delta/2: max|D[i].e| <= m*(p-1)*kErrorBound, so with p=256 and
// kErrorBound=16 this holds for m up to ~2000 (Delta/2 = 2^23).
inline constexpr int kErrorBound = 16;

// One CSPRNG byte (OpenSSL RAND_bytes).
inline std::uint8_t CsprngByte() {
  unsigned char b = 0;
  const int ok = RAND_bytes(&b, 1);
  assert(ok == 1 && "RAND_bytes failed");
  (void)ok;
  return static_cast<std::uint8_t>(b);
}

// Uniform ternary {0,1,2} via rejection (256 % 3 == 1; reject the top value to
// avoid bias).
inline std::uint64_t SampleTernary() {
  for (;;) {
    const std::uint8_t b = CsprngByte();
    if (b < 255) return static_cast<std::uint64_t>(b % 3);
  }
}

// Uniform LWE error in [-kErrorBound, kErrorBound] via rejection.
inline int SampleError() {
  const int range = 2 * kErrorBound + 1;       // 33
  const int limit = (256 / range) * range;     // 231
  for (;;) {
    const std::uint8_t b = CsprngByte();
    if (b < limit) return static_cast<int>(b % range) - kErrorBound;
  }
}

// One 64-bit step of splitmix64 (deterministic PRG for the PUBLIC matrix A).
inline std::uint64_t SplitMix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

class LheSimplePir {
 public:
  // db: m*m bytes row-major (D[i][j] = db[i*m + j]); short input is zero-padded.
  // n = secret dimension (keep n <= ~2000 so each limb partial stays < 65537).
  LheSimplePir(const std::vector<std::uint8_t>& db, std::uint64_t m,
               std::uint64_t n, std::uint64_t seed)
      : m_(m), n_(n) {
    d_.assign(m_ * m_, 0);
    for (std::size_t i = 0; i < db.size() && i < d_.size(); ++i) d_[i] = db[i];

    // A is m x n in Z_q, deterministic from seed.
    a_.resize(m_ * n_);
    for (std::uint64_t j = 0; j < m_; ++j)
      for (std::uint64_t k = 0; k < n_; ++k)
        a_[j * n_ + k] =
            static_cast<std::uint32_t>(SplitMix64(seed + j * n_ + k));

    // Ternary secret s in {0,1,2}^n, sampled from a CSPRNG (OpenSSL RAND_bytes),
    // fresh per client -- with the LWE error below this hides the queried index.
    s_.resize(n_);
    for (std::uint64_t k = 0; k < n_; ++k) s_[k] = SampleTernary();

    // Hint H = D * A  (m x n), mod 2^32.
    h_.assign(m_ * n_, 0);
    for (std::uint64_t i = 0; i < m_; ++i)
      for (std::uint64_t j = 0; j < m_; ++j) {
        const std::uint64_t dij = d_[i * m_ + j];
        if (dij == 0) continue;
        for (std::uint64_t k = 0; k < n_; ++k)
          h_[i * n_ + k] = static_cast<std::uint32_t>(
              h_[i * n_ + k] + dij * a_[j * n_ + k]);
      }

    // interm = H * s, recovered via the LHE path (query-independent, cached).
    std::vector<CipherBlob> enc_sk;
    const KeyBlob key = EncryptSecret(params_, s_, &enc_sk);
    const HintDecomp hd = DecomposeHint(params_, h_, m_, n_, kSimplePirElemBits);
    const std::vector<std::vector<CipherBlob>> hint_cts =
        ApplyHint(params_, hd, enc_sk);
    interm_ = RecoverAS(params_, key, hint_cts, m_, kSimplePirElemBits);
  }

  // Retrieve D[row][col] (the byte at db[row*m + col]).
  std::uint8_t Retrieve(std::uint64_t row, std::uint64_t col) const {
    // qu = A*s + Delta * u_col   (length m, mod 2^32).
    std::vector<std::uint32_t> qu(m_, 0);
    for (std::uint64_t j = 0; j < m_; ++j) {
      std::uint64_t acc = 0;
      for (std::uint64_t k = 0; k < n_; ++k) acc += a_[j * n_ + k] * s_[k];
      // + fresh bounded LWE error (makes the query IND-CPA); wraps mod 2^32.
      std::uint32_t qj = static_cast<std::uint32_t>(acc);
      qj += static_cast<std::uint32_t>(static_cast<std::int32_t>(SampleError()));
      qu[j] = qj;
    }
    qu[col] = static_cast<std::uint32_t>(qu[col] + kSimplePirDelta);

    // ans[row] = D[row] . qu   (mod 2^32).
    std::uint64_t ans = 0;
    for (std::uint64_t j = 0; j < m_; ++j)
      ans += static_cast<std::uint64_t>(d_[row * m_ + j]) * qu[j];
    const std::uint32_t ans_row = static_cast<std::uint32_t>(ans);

    // D[row][col] = round((ans - interm) / Delta) mod p.
    const std::uint32_t diff =
        static_cast<std::uint32_t>(ans_row - static_cast<std::uint32_t>(interm_[row]));
    const std::uint64_t rounded =
        (static_cast<std::uint64_t>(diff) + kSimplePirDelta / 2) / kSimplePirDelta;
    return static_cast<std::uint8_t>(rounded % kSimplePirP);
  }

  std::uint64_t m() const { return m_; }
  std::uint64_t n() const { return n_; }

 private:
  Params params_;
  std::uint64_t m_, n_;
  std::vector<std::uint64_t> d_;   // DB (m*m), entries in [0,256)
  std::vector<std::uint32_t> a_;   // A (m*n) in Z_q
  std::vector<std::uint64_t> s_;   // ternary secret (n)
  std::vector<std::uint64_t> h_;   // hint H = D*A (m*n), stored mod 2^32
  std::vector<std::uint64_t> interm_;  // H*s (m), recovered via LHE
};

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LHE_PIR_H_
