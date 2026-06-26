/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * LHE hint decompose / apply / recover, ported from underhood/underhood/hint.go
 * + server.go's HintAnswer (tiptoe chunk 1.1e). This is the homomorphic core of
 * Tiptoe's offline phase:
 *   - DecomposeHint: split the SimplePIR hint H into 4-bit limb planes, packed
 *     into NTT BFV plaintexts (n hint rows per plaintext coefficient block).
 *   - ApplyHint: given the client's per-element encrypted secret, compute the
 *     encrypted H*s for each limb via set_inner_product (server side).
 *   - RecoverAS: decrypt each limb's ciphertexts, center (FromModuloP), scale by
 *     the limb position, and sum -> H*s as T-width values (client side).
 *
 * The hint is passed as a flat row-major vector (hint[r*cols + c]); binding to
 * primihub's SimplePIR core::Matrix happens in chunk 1.1f. Real mode only
 * (pulls in @underhood//:rlwe via tiptoe_params.h). See
 * docs/pir/tiptoe-port-plan.md.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_HINT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_HINT_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_client.h"  // blobs
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_limb.h"
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_params.h"
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_secret.h"

namespace primihub::pir::tiptoe {

// Limb-decomposed hint: pts[b] holds rows*cols NTT plaintexts for limb plane b,
// indexed [r*cols + c]. Owns the rlwe plaintext objects (frees them on destroy).
class HintDecomp {
 public:
  HintDecomp() = default;
  ~HintDecomp() {
    for (auto& plane : pts)
      for (plaintext_t* pt : plane)
        if (pt != nullptr) plaintext_free(pt);
  }
  HintDecomp(const HintDecomp&) = delete;
  HintDecomp& operator=(const HintDecomp&) = delete;
  HintDecomp(HintDecomp&&) noexcept = default;
  HintDecomp& operator=(HintDecomp&&) noexcept = default;

  std::uint64_t hint_rows = 0;  // actual hint rows
  std::uint64_t rows = 0;       // ceil(hint_rows / N) row-blocks
  std::uint64_t cols = 0;       // secret dimension
  std::vector<std::vector<plaintext_t*>> pts;  // [limb][rows*cols]
};

namespace internal {

// Pack the index-th limb plane of the hint into rows*cols NTT plaintexts
// (underhood makePlaintext). hint[r*cols + c] is element (r, c).
inline std::vector<plaintext_t*> MakeLimbPlaintexts(
    const Params& p, const std::vector<std::uint64_t>& hint,
    std::uint64_t hint_rows, std::uint64_t cols, int index) {
  const std::uint64_t n = p.N();
  const std::uint64_t rows = (hint_rows + n - 1) / n;
  std::vector<plaintext_t*> out(rows * cols);
  for (plaintext_t*& o : out) o = plaintext_new();

  std::vector<std::uint64_t> vals(n);
  for (std::uint64_t c = 0; c < cols; ++c) {
    for (std::uint64_t r = 0; r < rows; ++r) {
      std::fill(vals.begin(), vals.end(), 0);
      for (std::uint64_t i = 0; i < n && (r * n + i) < hint_rows; ++i) {
        const std::uint64_t v = hint[(r * n + i) * cols + c];
        vals[i] = GetChunk(v, index);
      }
      plaintext_t* pt = out[r * cols + c];
      plaintext_set(pt, p.ctx(), vals.data(), n);
      plaintext_to_NTT(pt, p.ctx());
    }
  }
  return out;
}

}  // namespace internal

// Decompose the SimplePIR hint into limb planes (underhood decomposeHint).
// elem_bits is the SimplePIR element width (32 or 64).
inline HintDecomp DecomposeHint(const Params& p,
                                const std::vector<std::uint64_t>& hint,
                                std::uint64_t hint_rows, std::uint64_t cols,
                                int elem_bits) {
  HintDecomp d;
  const std::uint64_t n = p.N();
  d.hint_rows = hint_rows;
  d.rows = (hint_rows + n - 1) / n;
  d.cols = cols;

  const int max_limbs = MaxLimbs(elem_bits);
  const int limbs = LimbsFor(elem_bits);
  assert(limbs > 0 && "elem_bits must be 32 or 64");
  d.pts.resize(limbs);
  for (int b = 0; b < limbs; ++b)
    d.pts[b] = internal::MakeLimbPlaintexts(p, hint, hint_rows, cols,
                                            max_limbs - b - 1);
  return d;
}

namespace internal {

// Encrypted H*s for one limb plane (underhood applyHintOnce, serial). Returns
// one ciphertext blob per row-block.
inline std::vector<CipherBlob> ApplyHintOnce(
    const Params& p, const HintDecomp& hint,
    std::vector<ciphertext_t*>* enc_sk, int chunk) {
  assert(static_cast<std::uint64_t>(enc_sk->size()) == hint.cols);
  const std::uint64_t rows = hint.rows;
  const std::uint64_t cols = hint.cols;
  std::vector<CipherBlob> out(rows);
  std::vector<plaintext_t*> slice(cols);
  for (std::uint64_t i = 0; i < rows; ++i) {
    for (std::uint64_t k = 0; k < cols; ++k)
      slice[k] = hint.pts[chunk][i * cols + k];
    ciphertext_t* ct = ciphertext_new();
    ciphertext_set_inner_product(p.ctx(), ct, enc_sk->data(), slice.data(),
                                 cols);
    const std::size_t sz = ciphertext_size(ct);
    CipherBlob blob(sz);
    ciphertext_store(ct, blob.data(), sz);
    out[i] = std::move(blob);
    ciphertext_free(ct);
  }
  return out;
}

}  // namespace internal

// Server-side HintAnswer: encrypted H*s for every limb (underhood applyHint /
// Server.HintAnswer). enc_sk_blobs are the client's per-element encrypted
// secret ciphertexts (length == hint.cols). Returns [limb][row-block] blobs.
inline std::vector<std::vector<CipherBlob>> ApplyHint(
    const Params& p, const HintDecomp& hint,
    const std::vector<CipherBlob>& enc_sk_blobs) {
  std::vector<ciphertext_t*> enc_sk(enc_sk_blobs.size());
  for (std::size_t i = 0; i < enc_sk_blobs.size(); ++i) {
    enc_sk[i] = ciphertext_new();
    ciphertext_load(p.ctx(), enc_sk[i],
                    const_cast<std::uint8_t*>(enc_sk_blobs[i].data()),
                    enc_sk_blobs[i].size());
  }
  const int limbs = static_cast<int>(hint.pts.size());
  std::vector<std::vector<CipherBlob>> out(limbs);
  for (int b = 0; b < limbs; ++b)
    out[b] = internal::ApplyHintOnce(p, hint, &enc_sk, b);
  for (ciphertext_t* c : enc_sk) ciphertext_free(c);
  return out;
}

namespace internal {

// Decrypt + center one limb plane (underhood recoverASonce). Returns the
// FromModuloP-centered coefficient per matrix row (as wrapped uint64).
inline std::vector<std::uint64_t> RecoverASOnce(
    const Params& p, skey_t* sk, const std::vector<CipherBlob>& cts,
    std::uint64_t matrix_rows) {
  const std::uint64_t n = p.N();
  const std::uint64_t P = p.P();
  std::vector<std::uint64_t> out(matrix_rows, 0);
  std::vector<std::uint64_t> vals(n);
  for (std::size_t i = 0; i < cts.size(); ++i) {
    ciphertext_t* c = ciphertext_new();
    ciphertext_load(p.ctx(), c, const_cast<std::uint8_t*>(cts[i].data()),
                    cts[i].size());
    plaintext_t* pt = plaintext_new();
    key_decrypt(sk, c, pt);
    std::fill(vals.begin(), vals.end(), 0);
    plaintext_dump(pt, vals.data(), n);
    for (std::uint64_t j = 0;
         j < n && (static_cast<std::uint64_t>(i) * n + j) < matrix_rows; ++j) {
      out[static_cast<std::uint64_t>(i) * n + j] =
          static_cast<std::uint64_t>(FromModuloP<std::int64_t>(P, vals[j]));
    }
    plaintext_free(pt);
    ciphertext_free(c);
  }
  return out;
}

}  // namespace internal

// Client-side recover of H*s from the HintAnswer (underhood recoverAS): decrypt
// each limb, center, scale by the limb position, and sum (mod 2^elem_bits).
// Returns matrix_rows values (the H*s correction the SimplePIR client subtracts).
inline std::vector<std::uint64_t> RecoverAS(
    const Params& p, const KeyBlob& outer_key,
    const std::vector<std::vector<CipherBlob>>& hint_cts,
    std::uint64_t matrix_rows, int elem_bits) {
  skey_t* sk = key_new(p.ctx());
  key_load(p.ctx(), sk, const_cast<std::uint8_t*>(outer_key.data()),
           outer_key.size());

  const int max_limbs = MaxLimbs(elem_bits);
  const std::uint64_t mask =
      (elem_bits == 64) ? ~std::uint64_t{0}
                        : ((std::uint64_t{1} << elem_bits) - 1);
  std::vector<std::uint64_t> out(matrix_rows, 0);
  const int limbs = static_cast<int>(hint_cts.size());
  for (int b = 0; b < limbs; ++b) {
    const std::vector<std::uint64_t> part =
        internal::RecoverASOnce(p, sk, hint_cts[b], matrix_rows);
    const std::uint64_t scale = std::uint64_t{1}
                                << (kBitsPerLimb * (max_limbs - b - 1));
    for (std::uint64_t r = 0; r < matrix_rows; ++r) {
      const std::uint64_t pv = part[r] & mask;
      out[r] = (out[r] + pv * scale) & mask;
    }
  }
  key_free(sk);
  return out;
}

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_HINT_H_
