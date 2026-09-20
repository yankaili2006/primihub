/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_chacha_test — byte-for-byte against rand_chacha 0.3 test vectors
 * (rand 0.8.5 rand_chacha/src/chacha.rs): the IETF
 * draft-nir-cfrg-chacha20-poly1305-04 vectors and the from_seed
 * construction value. If these pass, the C++ stream equals
 * rand_chacha::ChaCha20Rng's exactly.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace primihub::pir::ypir {
namespace {

TEST(YpirChaChaTest, TrueValuesA_ZeroSeed_Blocks0And1) {
  std::array<std::uint8_t, 32> seed{};  // all zero
  auto rng = ChaChaRng::FromSeed(seed);  // ChaCha20
  std::uint32_t got[16];
  for (auto& g : got) g = rng.NextU32();
  const std::uint32_t block0[16] = {
      0xade0b876, 0x903df1a0, 0xe56a5d40, 0x28bd8653, 0xb819d2bd, 0x1aed8da0,
      0xccef36a8, 0xc70d778b, 0x7c5941da, 0x8d485751, 0x3fe02477, 0x374ad8b8,
      0xf4b8436a, 0x1ca11815, 0x69b687c3, 0x8665eeb2};
  for (int i = 0; i < 16; ++i) EXPECT_EQ(got[i], block0[i]) << "block0[" << i << "]";

  for (auto& g : got) g = rng.NextU32();
  const std::uint32_t block1[16] = {
      0xbee7079f, 0x7a385155, 0x7c97ba98, 0x0d082d73, 0xa0290fcb, 0x6965e348,
      0x3e53c612, 0xed7aee32, 0x7621b729, 0x434ee69c, 0xb03371d5, 0xd539d874,
      0x281fed31, 0x45fb0a51, 0x1f0ae1ac, 0x6f4d794b};
  for (int i = 0; i < 16; ++i) EXPECT_EQ(got[i], block1[i]) << "block1[" << i << "]";
}

TEST(YpirChaChaTest, TrueValuesB_Block1AfterSkippingBlock0) {
  std::array<std::uint8_t, 32> seed{};
  seed[31] = 1;
  auto rng = ChaChaRng::FromSeed(seed);
  for (int i = 0; i < 16; ++i) rng.NextU32();  // skip block 0
  std::uint32_t got[16];
  for (auto& g : got) g = rng.NextU32();
  const std::uint32_t expected[16] = {
      0x2452eb3a, 0x9249f8ec, 0x8d829d9b, 0xddd4ceb1, 0xe8252083, 0x60818b01,
      0xf38422b8, 0x5aaa49c9, 0xbb00ca8e, 0xda3ba7b4, 0xc4b592d1, 0xfdf2732f,
      0x4436274e, 0x2561b3c8, 0xebdd4aa6, 0xa0136c00};
  for (int i = 0; i < 16; ++i) EXPECT_EQ(got[i], expected[i]) << "[" << i << "]";
}

TEST(YpirChaChaTest, Construction_FromSeedFirstWord) {
  const std::array<std::uint8_t, 32> seed = {
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
      2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0};
  auto rng = ChaChaRng::FromSeed(seed);
  EXPECT_EQ(rng.NextU32(), 137206642u);
}

TEST(YpirChaChaTest, NextU64_LowWordFirst) {
  std::array<std::uint8_t, 32> seed{};
  auto a = ChaChaRng::FromSeed(seed);
  auto b = ChaChaRng::FromSeed(seed);
  const std::uint64_t u64 = a.NextU64();
  const std::uint64_t lo = b.NextU32();
  const std::uint64_t hi = b.NextU32();
  EXPECT_EQ(u64, (hi << 32) | lo);
  EXPECT_EQ(static_cast<std::uint32_t>(u64), 0xade0b876u);  // first keystream word
}

TEST(YpirChaChaTest, ChaCha12_DiffersFromChaCha20) {
  std::array<std::uint8_t, 32> seed{};
  auto rng20 = ChaChaRng::FromSeed(seed, 20);
  auto rng12 = ChaChaRng::FromSeed(seed, 12);
  EXPECT_EQ(rng20.NextU32(), 0xade0b876u);
  EXPECT_NE(rng12.NextU32(), 0xade0b876u);  // 12 rounds -> different stream
}

}  // namespace
}  // namespace primihub::pir::ypir
