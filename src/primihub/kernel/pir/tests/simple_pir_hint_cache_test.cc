/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimpleHintGen + SimpleHintCache unit tests. Mirrors DoublePIR's
 * hint_cache_test (chunk 2) — sibling LRU pattern, same fingerprint
 * scheme.
 */
#include "src/primihub/kernel/pir/operator/simple_pir/hint_cache.h"

#include <gtest/gtest.h>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::simple_pir {
namespace {

void MakeFixture(core::Database* db, core::LweParams* params) {
  params->n = 1024;
  params->logq = 32;
  std::string err;
  ASSERT_EQ(params->Pick(/*doublepir=*/false,
                          /*samples=*/static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS) << err;
  uint64_t l = 0, m = 0;
  ASSERT_EQ(core::ApproxSquareDatabaseDims(/*num=*/64, /*row_length=*/8,
                                            params->p, &l, &m, &err),
            retcode::SUCCESS) << err;
  params->l = l; params->m = m;
  ASSERT_EQ(db->SetupShape(/*num=*/64, /*row_length=*/8, *params, &err),
            retcode::SUCCESS) << err;
  for (uint64_t i = 0; i < params->l; ++i) {
    for (uint64_t j = 0; j < params->m; ++j) {
      db->mutable_data().Set(i, j,
                              static_cast<uint32_t>((i * 7 + j) & 0xFF));
    }
  }
  // NB: SimplePIR's Setup expects centered representation; the
  // caller-side ScalarSub(p/2) is done by SimplePirOperator before
  // calling GetOrComputeHint. The fingerprint test below is content-
  // sensitive so we apply the shift here for parity with the operator.
  db->mutable_data().ScalarSub(static_cast<uint32_t>(params->p / 2));
}

TEST(SimpleHintCacheFingerprintTest, IsStableAcrossIdenticalDbs) {
  core::Database db1, db2;
  core::LweParams p1, p2;
  MakeFixture(&db1, &p1);
  MakeFixture(&db2, &p2);
  EXPECT_EQ(FingerprintDb(db1, p1), FingerprintDb(db2, p2));
}

TEST(SimpleHintCacheFingerprintTest, DiffersWhenContentChanges) {
  core::Database db1, db2;
  core::LweParams p1, p2;
  MakeFixture(&db1, &p1);
  MakeFixture(&db2, &p2);
  db2.mutable_data().Set(0, 0, db2.mutable_data().Get(0, 0) + 1);
  EXPECT_NE(FingerprintDb(db1, p1), FingerprintDb(db2, p2));
}

TEST(SimpleHintCacheBasicTest, TryGetMissThenPutThenHit) {
  SimpleHintCache::Instance().Clear();
  SimpleHintCache& c = SimpleHintCache::Instance();
  c.SetCapacityForTest(4);

  SimplePirHint placeholder;
  EXPECT_FALSE(c.TryGet(0xfeedface, &placeholder));
  EXPECT_EQ(c.Misses(), 1u);
  EXPECT_EQ(c.Hits(), 0u);

  SimplePirHint h;
  h.A = core::Matrix::Zeros(2, 2);
  h.A.Set(0, 0, 11);
  c.Put(0xfeedface, h);
  EXPECT_EQ(c.Size(), 1u);

  SimplePirHint out;
  ASSERT_TRUE(c.TryGet(0xfeedface, &out));
  EXPECT_EQ(c.Hits(), 1u);
  EXPECT_EQ(out.A.Get(0, 0), 11u);
}

TEST(SimpleHintCacheBasicTest, LRUEvictsOldestBeyondCapacity) {
  SimpleHintCache::Instance().Clear();
  SimpleHintCache& c = SimpleHintCache::Instance();
  c.SetCapacityForTest(3);
  for (uint64_t k = 0; k < 5; ++k) {
    SimplePirHint h;
    h.A = core::Matrix::Zeros(1, 1);
    h.A.Set(0, 0, static_cast<uint32_t>(k));
    c.Put(k, std::move(h));
  }
  EXPECT_EQ(c.Size(), 3u);
  SimplePirHint out;
  EXPECT_FALSE(c.TryGet(0, &out));
  EXPECT_FALSE(c.TryGet(1, &out));
  EXPECT_TRUE(c.TryGet(2, &out));
  EXPECT_TRUE(c.TryGet(3, &out));
  EXPECT_TRUE(c.TryGet(4, &out));
}

TEST(GetOrComputeSimpleHintTest, MissThenHit) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge to populate the hint";
  }
  SimpleHintCache::Instance().Clear();

  core::Database db1;
  core::LweParams params;
  MakeFixture(&db1, &params);
  SimplePirHint hint1;
  SimpleHintGenStats stats1;
  bool hit1 = false;
  std::string err;
  ASSERT_EQ(GetOrComputeHint(&db1, params, &hint1, &err, &stats1, &hit1),
            retcode::SUCCESS) << err;
  EXPECT_FALSE(hit1);
  EXPECT_GT(stats1.setup_ms, 0.0);
  EXPECT_EQ(SimpleHintCache::Instance().Hits(), 0u);
  EXPECT_EQ(SimpleHintCache::Instance().Misses(), 1u);

  core::Database db2;
  core::LweParams params2;
  MakeFixture(&db2, &params2);
  SimplePirHint hint2;
  SimpleHintGenStats stats2;
  bool hit2 = false;
  ASSERT_EQ(GetOrComputeHint(&db2, params2, &hint2, &err, &stats2, &hit2),
            retcode::SUCCESS) << err;
  EXPECT_TRUE(hit2);
  EXPECT_DOUBLE_EQ(stats2.setup_ms, 0.0);
  EXPECT_EQ(SimpleHintCache::Instance().Hits(), 1u);

  // Hint payload identical on a cache hit.
  EXPECT_EQ(hint1.A.rows(), hint2.A.rows());
  EXPECT_EQ(hint1.A.cols(), hint2.A.cols());
  EXPECT_EQ(hint1.A.Get(0, 0), hint2.A.Get(0, 0));
  EXPECT_EQ(hint1.H.rows(), hint2.H.rows());
  EXPECT_EQ(hint1.H.cols(), hint2.H.cols());
  EXPECT_EQ(hint1.H.Get(0, 0), hint2.H.Get(0, 0));
}

}  // namespace
}  // namespace primihub::pir::simple_pir
