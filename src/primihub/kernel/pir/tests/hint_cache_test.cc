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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>

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

// ----- Persistence tests (task 5.6 chunk 4) ---------------------------

// RAII tmpfile path: mkstemp under /tmp, fd closed immediately (we just
// want a unique path), file unlinked on destruction.
class TmpPath {
 public:
  TmpPath() {
    char tmpl[] = "/tmp/hint_cache_test_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd >= 0) {
      ::close(fd);
      path_ = tmpl;
    }
  }
  ~TmpPath() {
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Build a small fake hint with a unique cell stamp so tests can
// distinguish entries.
DoublePirHint MakeFakeHint(uint32_t stamp) {
  DoublePirHint h;
  h.A1 = core::Matrix::Zeros(2, 2);
  h.A1.Set(0, 0, stamp);
  h.A1.Set(1, 1, stamp + 1);
  h.A2 = core::Matrix::Zeros(1, 3);
  h.A2.Set(0, 0, stamp + 100);
  h.H1_squished = core::Matrix::Zeros(1, 1);
  h.H1_squished.Set(0, 0, stamp + 200);
  h.A2_copy_transposed = core::Matrix::Zeros(3, 1);
  h.A2_copy_transposed.Set(0, 0, stamp + 300);
  h.H2_msg = core::Matrix::Zeros(1, 2);
  h.H2_msg.Set(0, 1, stamp + 400);
  h.info_after_setup.num = stamp;
  h.info_after_setup.p = 929;
  h.info_after_setup.logq = 32;
  return h;
}

TEST(HintCachePersistTest, SaveLoadRoundTripsThreeEntries) {
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());

  HintCache::Instance().Clear();
  HintCache::Instance().SetCapacityForTest(8);
  HintCache::Instance().Put(0x111, MakeFakeHint(1));
  HintCache::Instance().Put(0x222, MakeFakeHint(2));
  HintCache::Instance().Put(0x333, MakeFakeHint(3));
  // After three Puts MRU=0x333, LRU=0x111.

  std::string err;
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;

  // Wipe and reload.
  HintCache::Instance().Clear();
  ASSERT_EQ(HintCache::Instance().Size(), 0u);
  ASSERT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  EXPECT_EQ(HintCache::Instance().Size(), 3u);

  DoublePirHint out;
  ASSERT_TRUE(HintCache::Instance().TryGet(0x111, &out));
  EXPECT_EQ(out.A1.Get(0, 0), 1u);
  EXPECT_EQ(out.info_after_setup.num, 1u);
  ASSERT_TRUE(HintCache::Instance().TryGet(0x222, &out));
  EXPECT_EQ(out.A1.Get(0, 0), 2u);
  ASSERT_TRUE(HintCache::Instance().TryGet(0x333, &out));
  EXPECT_EQ(out.A1.Get(0, 0), 3u);
}

TEST(HintCachePersistTest, SaveEmptyThenLoadIsClean) {
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());

  HintCache::Instance().Clear();
  std::string err;
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;

  // File contains exactly the 16-byte cache header (PHHC + u16+u16 + u64=0).
  std::ifstream in(tmp.path(), std::ios::binary);
  ASSERT_TRUE(in.is_open());
  std::string data((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  EXPECT_EQ(data.size(), 16u);
  EXPECT_EQ(data.substr(0, 4), "PHHC");

  HintCache::Instance().Put(0xdead, MakeFakeHint(7));
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
  ASSERT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  // Load with empty file should leave the cache empty (replaces, not
  // merges).
  EXPECT_EQ(HintCache::Instance().Size(), 0u);
}

TEST(HintCachePersistTest, BadMagicRejectedNoMutation) {
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());
  // Create a file with a bad magic but otherwise valid framing.
  {
    std::ofstream out(tmp.path(), std::ios::binary | std::ios::trunc);
    const char bad[] = "XXXX\x01\x00\x00\x00";
    out.write(bad, 8);
    char zeros[8] = {0};
    out.write(zeros, 8);
  }
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0xaaa, MakeFakeHint(11));
  std::string err;
  EXPECT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::FAIL);
  EXPECT_NE(err.find("magic"), std::string::npos) << err;
  // Cache state preserved on FAIL.
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
  DoublePirHint out;
  ASSERT_TRUE(HintCache::Instance().TryGet(0xaaa, &out));
  EXPECT_EQ(out.A1.Get(0, 0), 11u);
}

TEST(HintCachePersistTest, VersionMismatchRejected) {
  TmpPath tmp;
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0xbeef, MakeFakeHint(42));
  std::string err;
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  // Read full content, bump version byte, write back.
  std::ifstream in(tmp.path(), std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  in.close();
  ASSERT_GE(data.size(), 6u);
  data[4] = static_cast<char>(0xfe);  // version low byte
  data[5] = static_cast<char>(0xff);  // version high byte
  std::ofstream out(tmp.path(), std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();

  EXPECT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::FAIL);
  EXPECT_NE(err.find("version"), std::string::npos) << err;
  // State preserved.
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
}

TEST(HintCachePersistTest, TruncatedFileRejected) {
  TmpPath tmp;
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0xcafe, MakeFakeHint(123));
  std::string err;
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  // Truncate body — keep header + entry header but drop the blob.
  std::ifstream in(tmp.path(), std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  in.close();
  ASSERT_GT(data.size(), 50u);
  data.resize(40);  // header (16) + first entry header (16) + 8 garbage
  std::ofstream out(tmp.path(), std::ios::binary | std::ios::trunc);
  out.write(data.data(), static_cast<std::streamsize>(data.size()));
  out.close();

  HintCache::Instance().Clear();
  HintCache::Instance().Put(0x999, MakeFakeHint(99));
  EXPECT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::FAIL);
  EXPECT_NE(err.find("truncated"), std::string::npos) << err;
  // State preserved (still has the post-clear entry).
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
}

TEST(HintCachePersistTest, OpenFailRejectedCleanly) {
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0xfeed, MakeFakeHint(55));
  std::string err;
  // Non-existent path with no rwx permission.
  const std::string bad_path = "/no/such/dir/hint_cache.bin";
  EXPECT_EQ(HintCache::Instance().LoadFromFile(bad_path, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("open"), std::string::npos) << err;
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
}

TEST(HintCachePersistTest, SaveAtomicReplacesExistingFile) {
  TmpPath tmp;
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0x1, MakeFakeHint(10));
  std::string err;
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  // Save again with different content; the old file should be replaced
  // atomically (no .tmp left behind).
  HintCache::Instance().Clear();
  HintCache::Instance().Put(0x2, MakeFakeHint(20));
  ASSERT_EQ(HintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  const std::string tmp_artifact = tmp.path() + ".tmp";
  std::ifstream check(tmp_artifact);
  EXPECT_FALSE(check.is_open()) << ".tmp artifact left behind";

  HintCache::Instance().Clear();
  ASSERT_EQ(HintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS)
      << err;
  EXPECT_EQ(HintCache::Instance().Size(), 1u);
  DoublePirHint out;
  EXPECT_FALSE(HintCache::Instance().TryGet(0x1, &out));
  EXPECT_TRUE(HintCache::Instance().TryGet(0x2, &out));
  EXPECT_EQ(out.A1.Get(0, 0), 20u);
}

}  // namespace
}  // namespace primihub::pir::double_pir
