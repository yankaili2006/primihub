/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"

namespace primihub::pir::ypir {

namespace {

inline std::uint32_t Rotl(std::uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

inline void QuarterRound(std::uint32_t& a, std::uint32_t& b, std::uint32_t& c,
                         std::uint32_t& d) {
  a += b; d ^= a; d = Rotl(d, 16);
  c += d; b ^= c; b = Rotl(b, 12);
  a += b; d ^= a; d = Rotl(d, 8);
  c += d; b ^= c; b = Rotl(b, 7);
}

inline std::uint32_t LeWord(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

// One 64-byte ChaCha block at the given counter/stream into out[16].
void Block(const std::uint32_t key[8], std::uint64_t counter,
           std::uint64_t stream, int rounds, std::uint32_t out[16]) {
  std::uint32_t s[16];
  s[0] = 0x61707865u; s[1] = 0x3320646eu;  // "expand 32-byte k"
  s[2] = 0x79622d32u; s[3] = 0x6b206574u;
  for (int i = 0; i < 8; ++i) s[4 + i] = key[i];
  s[12] = static_cast<std::uint32_t>(counter & 0xffffffffu);
  s[13] = static_cast<std::uint32_t>(counter >> 32);
  s[14] = static_cast<std::uint32_t>(stream & 0xffffffffu);
  s[15] = static_cast<std::uint32_t>(stream >> 32);

  std::uint32_t w[16];
  for (int i = 0; i < 16; ++i) w[i] = s[i];
  for (int r = 0; r < rounds; r += 2) {
    // column round
    QuarterRound(w[0], w[4], w[8], w[12]);
    QuarterRound(w[1], w[5], w[9], w[13]);
    QuarterRound(w[2], w[6], w[10], w[14]);
    QuarterRound(w[3], w[7], w[11], w[15]);
    // diagonal round
    QuarterRound(w[0], w[5], w[10], w[15]);
    QuarterRound(w[1], w[6], w[11], w[12]);
    QuarterRound(w[2], w[7], w[8], w[13]);
    QuarterRound(w[3], w[4], w[9], w[14]);
  }
  for (int i = 0; i < 16; ++i) out[i] = w[i] + s[i];
}

}  // namespace

ChaChaRng ChaChaRng::FromSeed(const std::array<std::uint8_t, 32>& seed,
                              int rounds) {
  ChaChaRng rng;
  for (int i = 0; i < 8; ++i) rng.key_[i] = LeWord(seed.data() + 4 * i);
  rng.rounds_ = rounds;
  rng.counter_ = 0;
  rng.stream_ = 0;
  rng.idx_ = 16;
  return rng;
}

void ChaChaRng::Refill() {
  Block(key_, counter_, stream_, rounds_, buf_);
  ++counter_;
  idx_ = 0;
}

std::uint32_t ChaChaRng::NextU32() {
  if (idx_ >= 16) Refill();
  return buf_[idx_++];
}

std::uint64_t ChaChaRng::NextU64() {
  const std::uint64_t lo = NextU32();
  const std::uint64_t hi = NextU32();
  return (hi << 32) | lo;
}

void ChaChaRng::FillBytes(std::uint8_t* out, std::size_t n) {
  std::size_t i = 0;
  while (i + 4 <= n) {
    const std::uint32_t w = NextU32();
    out[i] = static_cast<std::uint8_t>(w);
    out[i + 1] = static_cast<std::uint8_t>(w >> 8);
    out[i + 2] = static_cast<std::uint8_t>(w >> 16);
    out[i + 3] = static_cast<std::uint8_t>(w >> 24);
    i += 4;
  }
  if (i < n) {
    std::uint32_t w = NextU32();
    for (; i < n; ++i) {
      out[i] = static_cast<std::uint8_t>(w);
      w >>= 8;
    }
  }
}

}  // namespace primihub::pir::ypir
