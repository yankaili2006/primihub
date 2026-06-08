/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * HintCache unit tests — covers fingerprint stability, LRU eviction,
 * hit/miss accounting, and the GetOrComputeHint wrapper's
 * cache-hit vs cache-miss behaviour.
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_cache.h"

#include <gtest/gtest.h>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {
namespace {

// Build a tiny populated DB + matching params suitable for the cache.
// Mirrors the EndToEnd test fixture but at the smallest sane shape.
void MakeFixture(core::Database* db, core::LweParams* params) {
  params->n = 1024;
  params->logq = 32;
  std::string err;
  ASSERT_EQ(params->Pick(true, static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS) << err;
  uint64_t l = 0, m = 0;
  ASSERT_EQ(core::ApproxSquareDatabaseDims(64, 8, params->p, &l, &m, &err),
            retcode::SUCCESS);
  params->l = l; params->m = m;
  ASSERT_EQ(db->SetupShape(64, 8, *params, &err), retcode::SUCCESS);
  for (uint64_t i = 0; i < params->l; ++i) {
    for (uint64_t j = 0; j < params->m; ++j) {
      db->mutable_data().Set(i, j,
                              static_cast<uint32_t>((i * 7 + j) & 0xFF));
    }
  }
  db->mutable_data().ScalarSub(static_cast<uint32_t>(params->p / 2));
}

TEST(HintCacheFingerprintTest, IsStableAcrossIdenticalDbs) {
  core::Database db1, db2;
  core::LweParams p1, p2;
  MakeFixture(&db1, &p1);
  MakeFixture(&db2, &p2);
  // Same shape + same content => same fingerprint.
  EXPECT_EQ(FingerprintDb(db1, p1), FingerprintDb(db2, p2));
}

TEST(HintCacheFingerprintTest, DiffersWhenContentChanges) {
  core::Database db1, db2;
  core::LweParams p1, p2;
  MakeFixture(&db1, &p1);
  MakeFixture(&db2, &p2);
  // Flip one byte in db2.
  db2.mutable_data().Set(0, 0, db2.mutable_data().Get(0, 0) + 1);
  EXPECT_NE(FingerprintDb(db1, p1), FingerprintDb(db2, p2));
}

TEST(HintCacheBasicTest, TryGetMissThenPutThenHit) {
  HintCache::Instance().Clear();
  HintCache& c = HintCache::Instance();
  c.SetCapacityForTest(4);

  DoublePirHint placeholder;
  EXPECT_FALSE(c.TryGet(0xdeadbeef, &placeholder));
  EXPECT_EQ(c.Misses(), 1u);
  EXPECT_EQ(c.Hits(), 0u);
  EXPECT_EQ(c.Size(), 0u);

  // Put a fake hint and retrieve.
  DoublePirHint h;
  h.A1 = core::Matrix::Zeros(2, 2);
  h.A1.Set(0, 0, 7);
  c.Put(0xdeadbeef, h);
  EXPECT_EQ(c.Size(), 1u);

  DoublePirHint out;
  ASSERT_TRUE(c.TryGet(0xdeadbeef, &out));
  EXPECT_EQ(c.Hits(), 1u);
  EXPECT_EQ(out.A1.Get(0, 0), 7u);
}

TEST(HintCacheBasicTest, LRUEvictsOldestBeyondCapacity) {
  HintCache::Instance().Clear();
  HintCache& c = HintCache::Instance();
  c.SetCapacityForTest(3);

  for (uint64_t k = 0; k < 5; ++k) {
    DoublePirHint h;
    h.A1 = core::Matrix::Zeros(1, 1);
    h.A1.Set(0, 0, static_cast<uint32_t>(k));
    c.Put(k, std::move(h));
  }
  EXPECT_EQ(c.Size(), 3u);

  // Keys 0, 1 should be evicted; keys 2, 3, 4 retained.
  DoublePirHint out;
  EXPECT_FALSE(c.TryGet(0, &out));
  EXPECT_FALSE(c.TryGet(1, &out));
  EXPECT_TRUE(c.TryGet(2, &out));
  EXPECT_TRUE(c.TryGet(3, &out));
  EXPECT_TRUE(c.TryGet(4, &out));
}

TEST(HintCacheBasicTest, GetPromotesToMRU) {
  HintCache::Instance().Clear();
  HintCache& c = HintCache::Instance();
  c.SetCapacityForTest(2);

  DoublePirHint h;
  h.A1 = core::Matrix::Zeros(1, 1);
  c.Put(1, h);
  c.Put(2, h);
  // Touch 1 → makes 1 MRU; 2 becomes LRU.
  DoublePirHint out;
  ASSERT_TRUE(c.TryGet(1, &out));
  // Insert 3 → evicts the LRU which is now 2.
  c.Put(3, h);
  EXPECT_TRUE(c.TryGet(1, &out));
  EXPECT_FALSE(c.TryGet(2, &out));  // evicted
  EXPECT_TRUE(c.TryGet(3, &out));
}

TEST(GetOrComputeHintTest, MissThenHit) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge to populate the hint";
  }
  HintCache::Instance().Clear();

  // First call — cache miss, hint computed.
  core::Database db1;
  core::LweParams params;
  MakeFixture(&db1, &params);
  DoublePirHint hint1;
  HintGenStats stats1;
  bool hit1 = false;
  std::string err;
  ASSERT_EQ(GetOrComputeHint(&db1, params, &hint1, &err, &stats1, &hit1),
            retcode::SUCCESS) << err;
  EXPECT_FALSE(hit1);
  EXPECT_GT(stats1.setup_ms, 0.0);  // Setup ran
  EXPECT_EQ(HintCache::Instance().Hits(), 0u);
  EXPECT_EQ(HintCache::Instance().Misses(), 1u);

  // Second call with identical fresh db — cache hit.
  core::Database db2;
  core::LweParams params2;
  MakeFixture(&db2, &params2);
  DoublePirHint hint2;
  HintGenStats stats2;
  bool hit2 = false;
  ASSERT_EQ(GetOrComputeHint(&db2, params2, &hint2, &err, &stats2, &hit2),
            retcode::SUCCESS) << err;
  EXPECT_TRUE(hit2);
  EXPECT_DOUBLE_EQ(stats2.setup_ms, 0.0);  // no Setup work done
  EXPECT_EQ(HintCache::Instance().Hits(), 1u);

  // Hint contents identical on a real hit (matrices match).
  EXPECT_EQ(hint1.A1.rows(), hint2.A1.rows());
  EXPECT_EQ(hint1.A1.cols(), hint2.A1.cols());
  // Spot-check one cell.
  EXPECT_EQ(hint1.A1.Get(0, 0), hint2.A1.Get(0, 0));
  EXPECT_EQ(hint1.H2_msg.rows(), hint2.H2_msg.rows());
  EXPECT_EQ(hint1.H2_msg.cols(), hint2.H2_msg.cols());
  EXPECT_EQ(hint1.H2_msg.Get(0, 0), hint2.H2_msg.Get(0, 0));
}

}  // namespace
}  // namespace primihub::pir::double_pir
