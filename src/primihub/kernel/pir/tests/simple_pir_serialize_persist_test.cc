/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePIR hint serialize + on-disk persistence + operator
 * integration tests. Mirrors the DoublePIR coverage from
 * hint_serialize_test (chunk 3) + hint_cache_test persistence
 * additions (chunk 4) + double_pir_test PersistsHintCacheAcrossInstances
 * (chunk 5).
 */
#include "src/primihub/kernel/pir/operator/simple_pir/hint_serialize.h"
#include "src/primihub/kernel/pir/operator/simple_pir/hint_cache.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir.h"

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::simple_pir {
namespace {

SimplePirHint MakeFakeHint(uint32_t stamp) {
  SimplePirHint h;
  h.A = core::Matrix::Zeros(3, 5);
  h.A.Set(0, 0, stamp);
  h.A.Set(2, 4, stamp + 1);
  h.H = core::Matrix::Zeros(3, 4);
  h.H.Set(1, 2, stamp + 100);
  h.info_after_squish.num = stamp;
  h.info_after_squish.p = 991;
  h.info_after_squish.logq = 32;
  h.info_after_squish.basis = 10;
  h.info_after_squish.squishing = 3;
  return h;
}

void ExpectHintsEqual(const SimplePirHint& a, const SimplePirHint& b) {
  auto eq = [](const core::Matrix& x, const core::Matrix& y) {
    if (x.rows() != y.rows() || x.cols() != y.cols()) return false;
    for (uint64_t i = 0; i < x.rows(); ++i)
      for (uint64_t j = 0; j < x.cols(); ++j)
        if (x.Get(i, j) != y.Get(i, j)) return false;
    return true;
  };
  EXPECT_TRUE(eq(a.A, b.A));
  EXPECT_TRUE(eq(a.H, b.H));
  EXPECT_EQ(a.info_after_squish.num, b.info_after_squish.num);
  EXPECT_EQ(a.info_after_squish.p, b.info_after_squish.p);
  EXPECT_EQ(a.info_after_squish.logq, b.info_after_squish.logq);
  EXPECT_EQ(a.info_after_squish.basis, b.info_after_squish.basis);
  EXPECT_EQ(a.info_after_squish.squishing, b.info_after_squish.squishing);
}

// ----- chunk 3: wire serialization -----

TEST(SimpleHintSerializeTest, RoundTripPreservesFields) {
  SimplePirHint src = MakeFakeHint(7);
  std::string blob;
  std::string err;
  ASSERT_EQ(SerializeHint(src, &blob, &err), retcode::SUCCESS) << err;
  EXPECT_GE(blob.size(), 120u);  // header + DBinfo + 2 matrix headers

  SimplePirHint dst;
  ASSERT_EQ(DeserializeHint(blob, &dst, &err), retcode::SUCCESS) << err;
  ExpectHintsEqual(src, dst);
}

TEST(SimpleHintSerializeTest, BadMagicRejected) {
  SimplePirHint src = MakeFakeHint(1);
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  blob[0] = 'X';
  SimplePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("magic"), std::string::npos) << err;
}

TEST(SimpleHintSerializeTest, VersionMismatchRejected) {
  SimplePirHint src = MakeFakeHint(2);
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  blob[4] = static_cast<char>(0xfe);
  blob[5] = static_cast<char>(0xff);
  SimplePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("version"), std::string::npos) << err;
}

TEST(SimpleHintSerializeTest, TrailingBytesRejected) {
  SimplePirHint src = MakeFakeHint(3);
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  blob.append(8, '\0');
  SimplePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("trailing"), std::string::npos) << err;
}

TEST(SimpleHintSerializeTest, NullOutputRejected) {
  SimplePirHint src = MakeFakeHint(4);
  std::string err;
  EXPECT_EQ(SerializeHint(src, nullptr, &err), retcode::FAIL);
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  EXPECT_EQ(DeserializeHint(blob, nullptr, &err), retcode::FAIL);
}

// ----- chunk 4: on-disk persistence -----

class TmpPath {
 public:
  TmpPath() {
    char tmpl[] = "/tmp/simple_hint_cache_test_XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd >= 0) {
      ::close(fd);
      path_ = tmpl;
      ::unlink(path_.c_str());
    }
  }
  ~TmpPath() {
    if (!path_.empty()) ::unlink(path_.c_str());
  }
  const std::string& path() const { return path_; }
 private:
  std::string path_;
};

TEST(SimpleHintCachePersistTest, SaveLoadRoundTripsThreeEntries) {
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());
  SimpleHintCache::Instance().Clear();
  SimpleHintCache::Instance().SetCapacityForTest(8);
  SimpleHintCache::Instance().Put(0x111, MakeFakeHint(1));
  SimpleHintCache::Instance().Put(0x222, MakeFakeHint(2));
  SimpleHintCache::Instance().Put(0x333, MakeFakeHint(3));

  std::string err;
  ASSERT_EQ(SimpleHintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS) << err;

  SimpleHintCache::Instance().Clear();
  ASSERT_EQ(SimpleHintCache::Instance().Size(), 0u);
  ASSERT_EQ(SimpleHintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(SimpleHintCache::Instance().Size(), 3u);

  SimplePirHint out;
  ASSERT_TRUE(SimpleHintCache::Instance().TryGet(0x111, &out));
  EXPECT_EQ(out.A.Get(0, 0), 1u);
  ASSERT_TRUE(SimpleHintCache::Instance().TryGet(0x222, &out));
  EXPECT_EQ(out.A.Get(0, 0), 2u);
  ASSERT_TRUE(SimpleHintCache::Instance().TryGet(0x333, &out));
  EXPECT_EQ(out.A.Get(0, 0), 3u);
}

TEST(SimpleHintCachePersistTest, BadMagicPreservesState) {
  TmpPath tmp;
  {
    std::ofstream out(tmp.path(), std::ios::binary | std::ios::trunc);
    const char bad[] = "ZZZZ\x01\x00\x00\x00\0\0\0\0\0\0\0\0";
    out.write(bad, sizeof(bad) - 1);
  }
  SimpleHintCache::Instance().Clear();
  SimpleHintCache::Instance().Put(0xaaa, MakeFakeHint(11));
  std::string err;
  EXPECT_EQ(SimpleHintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::FAIL);
  EXPECT_NE(err.find("magic"), std::string::npos) << err;
  EXPECT_EQ(SimpleHintCache::Instance().Size(), 1u);
  SimplePirHint out;
  ASSERT_TRUE(SimpleHintCache::Instance().TryGet(0xaaa, &out));
  EXPECT_EQ(out.A.Get(0, 0), 11u);
}

TEST(SimpleHintCachePersistTest, SaveAtomicReplacesExistingFile) {
  TmpPath tmp;
  SimpleHintCache::Instance().Clear();
  SimpleHintCache::Instance().Put(0x1, MakeFakeHint(10));
  std::string err;
  ASSERT_EQ(SimpleHintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS) << err;
  SimpleHintCache::Instance().Clear();
  SimpleHintCache::Instance().Put(0x2, MakeFakeHint(20));
  ASSERT_EQ(SimpleHintCache::Instance().SaveToFile(tmp.path(), &err),
            retcode::SUCCESS) << err;
  std::ifstream check(tmp.path() + ".tmp");
  EXPECT_FALSE(check.is_open()) << ".tmp artifact left behind";

  SimpleHintCache::Instance().Clear();
  ASSERT_EQ(SimpleHintCache::Instance().LoadFromFile(tmp.path(), &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(SimpleHintCache::Instance().Size(), 1u);
  SimplePirHint out;
  EXPECT_FALSE(SimpleHintCache::Instance().TryGet(0x1, &out));
  EXPECT_TRUE(SimpleHintCache::Instance().TryGet(0x2, &out));
}

// ----- chunk 5: operator integration -----

void RunOnceWithHintPath(const std::string& hint_path) {
  std::vector<std::string> db;
  db.reserve(64);
  for (uint64_t i = 0; i < 64; ++i) {
    db.push_back(std::to_string((i * 13 + 7) & 0xFF));
  }
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  opt.hint_path = hint_path;
  SimplePirOperator op(opt);
  PirDataType input;
  input["db_content"] = db;
  input["query_indices"] = {"0", "27", "63"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
}

TEST(SimplePirOperatorPersistTest, PersistsHintCacheAcrossInstances) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());

  SimpleHintCache::Instance().Clear();
  const uint64_t hits0 = SimpleHintCache::Instance().Hits();
  const uint64_t misses0 = SimpleHintCache::Instance().Misses();
  RunOnceWithHintPath(tmp.path());
  EXPECT_EQ(SimpleHintCache::Instance().Hits(), hits0);
  EXPECT_EQ(SimpleHintCache::Instance().Misses(), misses0 + 1u);
  std::ifstream check(tmp.path(), std::ios::binary);
  ASSERT_TRUE(check.is_open()) << "SaveToFile did not create " << tmp.path();
  check.close();

  SimpleHintCache::Instance().Clear();
  const uint64_t hits1 = SimpleHintCache::Instance().Hits();
  const uint64_t misses1 = SimpleHintCache::Instance().Misses();
  RunOnceWithHintPath(tmp.path());
  EXPECT_EQ(SimpleHintCache::Instance().Hits(), hits1 + 1u);
  EXPECT_EQ(SimpleHintCache::Instance().Misses(), misses1);
}

TEST(SimplePirOperatorPersistTest, MissingHintFileDegradesGracefully) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  TmpPath tmp;
  ASSERT_FALSE(tmp.path().empty());
  SimpleHintCache::Instance().Clear();
  RunOnceWithHintPath(tmp.path());
  std::ifstream check(tmp.path(), std::ios::binary);
  EXPECT_TRUE(check.is_open());
}

}  // namespace
}  // namespace primihub::pir::simple_pir
