/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_prng_test — chunk 2b-i tests. Validates the SeededRng
 * abstraction's determinism + cross-seed independence. Does NOT
 * cross-check against rand_chacha::ChaCha12Rng since the chunk-
 * 2b-i backing engine is mt19937_64 and intentionally produces
 * a different stream; that cross-check moves to chunk 2b-iii's
 * native ChaCha12 port if ever needed.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

#include <array>
#include <cstdint>
#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::frodo {
namespace {

SeedBytes MakeSeed(std::uint8_t fill) {
  SeedBytes s;
  s.fill(fill);
  return s;
}

SeedBytes MakeIotaSeed(std::uint8_t start) {
  SeedBytes s;
  for (std::size_t i = 0; i < 32; ++i) {
    s[i] = static_cast<std::uint8_t>(start + i);
  }
  return s;
}

TEST(FrodoSeededRngTest, SameSeed_SameStream) {
  // Determinism guarantee that callers (notably
  // GenerateLweMatrixFromSeed) depend on for client/server matrix A
  // agreement.
  const auto seed = MakeIotaSeed(/*start=*/0);
  SeededRng a(seed), b(seed);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(a.NextU32(), b.NextU32())
        << "diverged at NextU32 call " << i;
  }
}

TEST(FrodoSeededRngTest, DifferentSeed_DifferentStream) {
  // Cross-seed independence: at least one of the first 16 u32s
  // must differ. mt19937_64 produces 64-bit blocks so we sample
  // enough to make a coincidental collision astronomically
  // unlikely (probability ≈ 16 · 2^-32).
  SeededRng a(MakeSeed(0x00));
  SeededRng b(MakeSeed(0xFF));
  bool any_diff = false;
  for (int i = 0; i < 16; ++i) {
    if (a.NextU32() != b.NextU32()) {
      any_diff = true;
    }
  }
  EXPECT_TRUE(any_diff)
      << "two seeds {0x00*32} and {0xFF*32} produced identical "
      << "first 16 u32s — extraordinarily unlikely";
}

TEST(FrodoSeededRngTest, NextU64_DifferentSeedsDiffer) {
  SeededRng a(MakeSeed(0x42));
  SeededRng b(MakeSeed(0x43));
  bool any_diff = false;
  for (int i = 0; i < 8; ++i) {
    if (a.NextU64() != b.NextU64()) any_diff = true;
  }
  EXPECT_TRUE(any_diff);
}

TEST(FrodoSeededRngTest, NextU64_SameSeedSameStream) {
  const auto seed = MakeIotaSeed(/*start=*/100);
  SeededRng a(seed), b(seed);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(a.NextU64(), b.NextU64()) << "diverged at NextU64 call " << i;
  }
}

TEST(FrodoSeededRngTest, EveryByteOfSeedMatters) {
  // Flipping any single byte of the seed must perturb the stream.
  // This guards against accidental seed-derivation shortcuts that
  // only mix the first few bytes (e.g., a future engine swap that
  // mis-derives the seed_seq).
  const auto base = MakeIotaSeed(0);
  SeededRng base_rng(base);
  const std::uint32_t base_first = base_rng.NextU32();
  std::set<std::uint32_t> first_outputs;
  first_outputs.insert(base_first);
  for (std::size_t i = 0; i < 32; ++i) {
    auto perturbed = base;
    perturbed[i] ^= 0x80;  // flip the high bit of byte i
    SeededRng pert_rng(perturbed);
    const std::uint32_t first = pert_rng.NextU32();
    EXPECT_NE(first, base_first)
        << "flipping byte " << i << " did not change the first u32 — "
        << "seed derivation may be ignoring this byte";
    first_outputs.insert(first);
  }
  // With 33 perturbations on a 32-bit output, collisions are
  // possible but most should be distinct.
  EXPECT_GT(first_outputs.size(), 16u)
      << "too many seed perturbations produced the same first u32 "
      << "(" << first_outputs.size() << " unique out of 33)";
}

TEST(FrodoSeededRngTest, AllZeroSeed_Works) {
  // mt19937 has a documented all-zero-state failure case, but
  // std::seed_seq mitigates it by mixing in additional entropy
  // sources. Spot-check that all-zero seed still produces a
  // non-zero first output (probability of legitimate-zero ≈ 2^-32).
  SeededRng z(MakeSeed(0x00));
  bool any_nonzero = false;
  for (int i = 0; i < 4; ++i) {
    if (z.NextU32() != 0) {
      any_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(any_nonzero)
      << "all-zero seed produced 4 consecutive zero u32s — "
      << "seed_seq may have failed to break the trivial state";
}

}  // namespace
}  // namespace primihub::pir::frodo
