/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Unit test for the pure limb helpers (tiptoe chunk 1.1e). Zero-dependency,
 * runs in the default .50 build.
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_limb.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace primihub::pir::tiptoe {
namespace {

TEST(TiptoeLimbTest, GetChunkExtractsNibbles) {
  // 0xDEADBEEF, limb 0 = 0xF, limb 1 = 0xE, limb 2 = 0xE, ... limb 7 = 0xD.
  const std::uint64_t v = 0xDEADBEEFull;
  EXPECT_EQ(GetChunk(v, 0), 0xFu);
  EXPECT_EQ(GetChunk(v, 1), 0xEu);
  EXPECT_EQ(GetChunk(v, 2), 0xEu);
  EXPECT_EQ(GetChunk(v, 3), 0xBu);
  EXPECT_EQ(GetChunk(v, 4), 0xDu);
  EXPECT_EQ(GetChunk(v, 5), 0xAu);
  EXPECT_EQ(GetChunk(v, 6), 0xEu);
  EXPECT_EQ(GetChunk(v, 7), 0xDu);
}

TEST(TiptoeLimbTest, GetChunkHighLimbs64Bit) {
  const std::uint64_t v = 0xF000000000000000ull;
  EXPECT_EQ(GetChunk(v, 15), 0xFu);
  EXPECT_EQ(GetChunk(v, 14), 0x0u);
  // Reconstruct: sum of limb*16^chunk == v.
  std::uint64_t recon = 0;
  for (int c = 0; c < 16; ++c)
    recon |= GetChunk(v, c) << (c * kBitsPerLimb);
  EXPECT_EQ(recon, v);
}

TEST(TiptoeLimbTest, LimbCounts) {
  EXPECT_EQ(MaxLimbs(64), 16);
  EXPECT_EQ(MaxLimbs(32), 8);
  EXPECT_EQ(LimbsFor(64), 8);
  EXPECT_EQ(LimbsFor(32), 5);
  EXPECT_EQ(LimbsFor(16), 0);  // unsupported
  EXPECT_EQ(kBitsPerLimb, 4);
}

}  // namespace
}  // namespace primihub::pir::tiptoe
