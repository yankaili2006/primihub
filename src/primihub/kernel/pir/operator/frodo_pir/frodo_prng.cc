/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * !! WARNING !! See frodo_prng.h header for the full security
 * regression analysis. Short version: this file backs SeededRng
 * with std::mt19937_64 which is NOT cryptographically secure.
 * Chunk 2b-ii will swap to OpenSSL ChaCha20 keeping the same API
 * surface. Do not deploy this version to production.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

#include <vector>

namespace primihub::pir::frodo {

namespace {

// Deterministic seed-derivation that uses every byte of the 32-byte
// seed. We pack the seed into 8 u32s (little-endian within each
// 4-byte chunk) and feed them through std::seed_seq to the engine.
// std::seed_seq is implementation-defined but reproducible within
// a fixed libstdc++ version; that's enough for our threat model
// while chunk 2b-ii is pending.
std::mt19937_64 MakeEngineFromSeed(const SeedBytes& seed) {
  std::vector<std::uint32_t> u32s(8, 0);
  for (std::size_t i = 0; i < 8; ++i) {
    std::uint32_t v = 0;
    for (std::size_t b = 0; b < 4; ++b) {
      v |= static_cast<std::uint32_t>(seed[i * 4 + b]) << (b * 8);
    }
    u32s[i] = v;
  }
  std::seed_seq ss(u32s.begin(), u32s.end());
  return std::mt19937_64(ss);
}

}  // namespace

SeededRng::SeededRng(const SeedBytes& seed)
    : engine_(MakeEngineFromSeed(seed)),
      buffer_(0),
      buffer_has_hi_(false) {}

std::uint32_t SeededRng::NextU32() {
  // Pull 64 bits, hand out the low 32, stash the high 32 for the
  // next call. Consecutive NextU32 calls thus return (lo, hi) of
  // the same u64 — equivalent to NextU64() split little-endian.
  if (!buffer_has_hi_) {
    buffer_ = engine_();
    buffer_has_hi_ = true;
    return static_cast<std::uint32_t>(buffer_ & 0xFFFFFFFFu);
  }
  buffer_has_hi_ = false;
  return static_cast<std::uint32_t>((buffer_ >> 32) & 0xFFFFFFFFu);
}

std::uint64_t SeededRng::NextU64() {
  // If there's a stashed high half from an odd NextU32 call, we
  // need to drain it first so NextU32/NextU64 mix coherently.
  if (buffer_has_hi_) {
    const std::uint32_t hi = static_cast<std::uint32_t>(buffer_ >> 32);
    buffer_has_hi_ = false;
    const std::uint64_t fresh = engine_();
    return (static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(fresh & 0xFFFFFFFFu))
            << 32) |
           hi;
  }
  return engine_();
}

}  // namespace primihub::pir::frodo
