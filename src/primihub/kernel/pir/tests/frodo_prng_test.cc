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



// ---- Chunk 2b-ii ChaCha20 vector pinning -----------------------

// RFC 8439 ChaCha20 Test Vector 1:
//   key   = 0x00 * 32
//   nonce = 0x00 * 12
//   counter = 0
//   first 64 keystream bytes:
//     76 b8 e0 ad a0 f1 3d 90 40 5d 6a e5 53 86 bd 28
//     bd d2 19 b8 a0 8d ed 1a a8 36 ef cc 8b 77 0d c7
//     da 41 59 7c 51 57 48 8d 77 24 e0 3f b8 d8 4a 37
//     6a 43 b8 f4 15 18 a1 1c c3 87 b6 69 b2 ee 65 86
//
// Our SeededRng with all-zero seed must produce this stream, read
// little-endian as u32s. NextU32 #1 = LE(76 b8 e0 ad) = 0xADE0B876.
// NextU32 #2 = LE(a0 f1 3d 90) = 0x903DF1A0. This is the byte-for-
// byte ChaCha20 contract — any divergence here means the cipher
// glue is wrong.

TEST(FrodoSeededRngTest, ChaCha20_RFC8439_TestVector1_FirstU32s) {
  SeededRng z(MakeSeed(0x00));
  const std::vector<std::uint32_t> rfc_first_16_u32 = {
      0xADE0B876u, 0x903DF1A0u, 0xE56A5D40u, 0x28BD8653u,
      0xB819D2BDu, 0x1AED8DA0u, 0xCCEF36A8u, 0xC70D778Bu,
      0x7C5941DAu, 0x8D485751u, 0x3FE02477u, 0x374AD8B8u,
      0xF4B8436Au, 0x1CA11815u, 0x69B687C3u, 0x8665EEB2u,
  };
  for (std::size_t i = 0; i < rfc_first_16_u32.size(); ++i) {
    const std::uint32_t got = z.NextU32();
    EXPECT_EQ(got, rfc_first_16_u32[i])
        << "ChaCha20 RFC 8439 test vector 1 mismatch at u32 index "
        << i << " — backing cipher diverges from ChaCha20 contract";
  }
}

TEST(FrodoSeededRngTest, ChaCha20_BlockBoundaryStraddle_NextU64) {
  // One ChaCha20 block = 64 bytes. Read 15 u32s = 60 bytes from
  // the first block, then a NextU64 reads 8 bytes straddling the
  // block boundary (4 from end of block 1 + 4 from start of block
  // 2). The straddling u64 must equal the RFC 8439 vector
  // concatenation of bytes 60..68:
  //   block 0 bytes 60..63 = c3 87 b6 69
  //   block 1 byte  0..3  = a series we cross-check by reading
  //                          via NextU32 from a fresh SeededRng
  //                          and concatenating.
  SeededRng a(MakeSeed(0x00));
  for (int i = 0; i < 15; ++i) {
    (void)a.NextU32();
  }
  const std::uint64_t straddle = a.NextU64();

  // Independent path: read 17 u32s from a fresh instance and
  // reconstruct the same u64 from u32 #15 lo32 + u32 #16 hi32 of
  // the byte stream.
  // Specifically straddle = LE(bytes 60..67) = LE(c3 87 b6 69 ?? ?? ?? ??)
  // We get bytes 60..67 as: u32 #15 (bytes 60..63) || first 4
  // bytes of next block.
  SeededRng b(MakeSeed(0x00));
  std::uint32_t got_u32s[17];
  for (int i = 0; i < 17; ++i) {
    got_u32s[i] = b.NextU32();
  }
  // bytes 60..67 = LE-bytes-of(got_u32s[15]) || LE-bytes-of(got_u32s[16])
  // straddle = LE u64 of those 8 bytes
  // = got_u32s[15] | (got_u32s[16] << 32)
  const std::uint64_t expected =
      static_cast<std::uint64_t>(got_u32s[15]) |
      (static_cast<std::uint64_t>(got_u32s[16]) << 32);
  EXPECT_EQ(straddle, expected)
      << "NextU64 across a ChaCha20 block boundary does not match "
      << "two NextU32 reads at the same byte offset";
}

TEST(FrodoSeededRngTest, ChaCha20_NotMt19937_LegacyComparison) {
  // Catches a regression where the swap to ChaCha20 silently fell
  // back to mt19937. mt19937_64 with seed_seq over 8 zeros produces
  // a specific first u32 (computed once on the chunk-2b-i toolchain
  // for documentation; not a tight assertion). The RFC value
  // 0xADE0B876 is distinctive enough that a chunk-2b-i-style engine
  // can't produce it from a zero seed.
  SeededRng z(MakeSeed(0x00));
  EXPECT_EQ(z.NextU32(), 0xADE0B876u)
      << "first u32 from all-zero seed should be the RFC 8439 "
      << "ChaCha20 vector; if not, the swap regressed to a "
      << "different engine";
}

}  // namespace
}  // namespace primihub::pir::frodo
