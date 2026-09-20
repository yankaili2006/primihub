/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_chacha — native ChaCha RNG byte-compatible with upstream
 * rand_chacha (rand 0.8 / rand_chacha 0.3). Chunk 2b-iii of the port
 * (see docs/pir/ypir-port-plan.md); the rand_chacha::ChaCha20Rng-
 * compatible PRNG that ypir lwe.rs (ChaCha20Rng) and the deferred
 * LWEClient::new key generation need, and that frodo's SeededRng (which
 * is OpenSSL ChaCha20, NOT rand_chacha-stream-compatible) can move to.
 *
 * Construction (matches rand_chacha 0.3 docs): 16-word state =
 *   [c0 c1 c2 c3]   ("expand 32-byte k" sigma constants)
 *   [k0 .. k7]      (32-byte seed, little-endian words)
 *   [ctr_lo ctr_hi] (64-bit block counter, starts 0)
 *   [str_lo str_hi] (64-bit stream id in place of a nonce, starts 0)
 * Each 64-byte block runs `rounds` ChaCha rounds (column+diagonal
 * double-rounds) and adds the initial state; next_u32 yields the 16
 * words in order, incrementing the counter per block. next_u64 reads
 * two consecutive words, low word first (rand_core BlockRng semantics).
 *
 * Verified byte-for-byte against rand_chacha's own test vectors
 * (IETF draft-nir-cfrg-chacha20-poly1305-04 vectors 1-4, and the
 * from_seed construction value).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CHACHA_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CHACHA_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace primihub::pir::ypir {

class ChaChaRng {
 public:
  // Build from a 32-byte seed. `rounds` defaults to 20 (ChaCha20Rng);
  // pass 12 for ChaCha12Rng (rand's StdRng) or 8 for ChaCha8Rng.
  static ChaChaRng FromSeed(const std::array<std::uint8_t, 32>& seed,
                            int rounds = 20);

  // Mirrors rand's RngCore::next_u32 over a rand_chacha core.
  std::uint32_t NextU32();
  // Mirrors RngCore::next_u64: (hi << 32) | lo with lo drawn first.
  std::uint64_t NextU64();
  // Fill `out[0..n)` with keystream bytes (little-endian words),
  // matching rand_core fill_bytes at word granularity.
  void FillBytes(std::uint8_t* out, std::size_t n);

 private:
  void Refill();

  std::uint32_t key_[8] = {0};
  std::uint64_t counter_ = 0;
  std::uint64_t stream_ = 0;
  int rounds_ = 20;
  std::uint32_t buf_[16] = {0};
  std::size_t idx_ = 16;  // == 16 forces a refill on first draw
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CHACHA_H_
