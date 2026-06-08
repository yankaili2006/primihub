/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SerializeHint / DeserializeHint round-trip + corruption coverage.
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"

#include <gtest/gtest.h>
#include <cstdint>
#include <string>

#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {
namespace {

// Build a fixture hint with distinct, deterministic values in each
// matrix so the round-trip can spot mis-ordering bugs.
DoublePirHint MakeFixture() {
  DoublePirHint h;
  auto fill = [](core::Matrix* m, uint32_t seed) {
    for (uint64_t i = 0; i < m->rows(); ++i) {
      for (uint64_t j = 0; j < m->cols(); ++j) {
        m->Set(i, j, seed + static_cast<uint32_t>(i * m->cols() + j));
      }
    }
  };
  h.A1 = core::Matrix::Zeros(3, 5);              fill(&h.A1, 1000);
  h.A2 = core::Matrix::Zeros(2, 7);              fill(&h.A2, 2000);
  h.H1_squished = core::Matrix::Zeros(4, 3);     fill(&h.H1_squished, 3000);
  h.A2_copy_transposed = core::Matrix::Zeros(7, 2);
  fill(&h.A2_copy_transposed, 4000);
  h.H2_msg = core::Matrix::Zeros(1, 6);          fill(&h.H2_msg, 5000);

  h.info_after_setup.num = 64;
  h.info_after_setup.row_length = 8;
  h.info_after_setup.packing = 1;
  h.info_after_setup.ne = 2;
  h.info_after_setup.x = 2;
  h.info_after_setup.p = 929;
  h.info_after_setup.logq = 32;
  h.info_after_setup.basis = 10;
  h.info_after_setup.squishing = 3;
  h.info_after_setup.cols = 3;
  return h;
}

void ExpectHintsEqual(const DoublePirHint& a, const DoublePirHint& b) {
  auto eq = [](const core::Matrix& x, const core::Matrix& y) {
    if (x.rows() != y.rows() || x.cols() != y.cols()) return false;
    for (uint64_t i = 0; i < x.rows(); ++i) {
      for (uint64_t j = 0; j < x.cols(); ++j) {
        if (x.Get(i, j) != y.Get(i, j)) return false;
      }
    }
    return true;
  };
  EXPECT_TRUE(eq(a.A1, b.A1));
  EXPECT_TRUE(eq(a.A2, b.A2));
  EXPECT_TRUE(eq(a.H1_squished, b.H1_squished));
  EXPECT_TRUE(eq(a.A2_copy_transposed, b.A2_copy_transposed));
  EXPECT_TRUE(eq(a.H2_msg, b.H2_msg));
  EXPECT_EQ(a.info_after_setup.num, b.info_after_setup.num);
  EXPECT_EQ(a.info_after_setup.row_length, b.info_after_setup.row_length);
  EXPECT_EQ(a.info_after_setup.packing, b.info_after_setup.packing);
  EXPECT_EQ(a.info_after_setup.ne, b.info_after_setup.ne);
  EXPECT_EQ(a.info_after_setup.x, b.info_after_setup.x);
  EXPECT_EQ(a.info_after_setup.p, b.info_after_setup.p);
  EXPECT_EQ(a.info_after_setup.logq, b.info_after_setup.logq);
  EXPECT_EQ(a.info_after_setup.basis, b.info_after_setup.basis);
  EXPECT_EQ(a.info_after_setup.squishing, b.info_after_setup.squishing);
  EXPECT_EQ(a.info_after_setup.cols, b.info_after_setup.cols);
}

TEST(HintSerializeTest, RoundTripPreservesAllFields) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  std::string err;
  ASSERT_EQ(SerializeHint(src, &blob, &err), retcode::SUCCESS) << err;
  EXPECT_GE(blob.size(), 168u);  // header + DBinfo + 5 matrix headers

  DoublePirHint dst;
  ASSERT_EQ(DeserializeHint(blob, &dst, &err), retcode::SUCCESS) << err;
  ExpectHintsEqual(src, dst);
}

TEST(HintSerializeTest, MagicCheckedFirst) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  // Flip the magic.
  blob[0] = 'X';
  DoublePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("magic"), std::string::npos) << err;
}

TEST(HintSerializeTest, VersionMismatchRejected) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  // Bump the version byte (offset 4 = low byte of uint16).
  blob[4] = static_cast<char>(0xfe);
  blob[5] = static_cast<char>(0xff);
  DoublePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("version"), std::string::npos) << err;
}

TEST(HintSerializeTest, NonzeroReservedRejected) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  // Reserved word lives at offset 6..7.
  blob[6] = 1;
  DoublePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("reserved"), std::string::npos) << err;
}

TEST(HintSerializeTest, TruncatedBlobRejected) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  // Drop the last 16 bytes — should land inside the last matrix's body.
  blob.resize(blob.size() - 16);
  DoublePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("truncated"), std::string::npos) << err;
}

TEST(HintSerializeTest, TrailingBytesRejected) {
  DoublePirHint src = MakeFixture();
  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  blob.append(8, '\0');
  DoublePirHint dst;
  std::string err;
  EXPECT_EQ(DeserializeHint(blob, &dst, &err), retcode::FAIL);
  EXPECT_NE(err.find("trailing"), std::string::npos) << err;
}

TEST(HintSerializeTest, NullOutputRejected) {
  DoublePirHint src = MakeFixture();
  std::string err;
  EXPECT_EQ(SerializeHint(src, nullptr, &err), retcode::FAIL);

  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob), retcode::SUCCESS);
  EXPECT_EQ(DeserializeHint(blob, nullptr, &err), retcode::FAIL);
}

TEST(HintSerializeTest, RealHintGenOutputRoundTrips) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs HintGen vendored kernel path";
  }
  core::LweParams params;
  params.n = 1024;
  params.logq = 32;
  std::string err;
  ASSERT_EQ(params.Pick(/*doublepir=*/true,
                         /*samples=*/static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS) << err;
  uint64_t l = 0, m = 0;
  ASSERT_EQ(core::ApproxSquareDatabaseDims(/*num=*/64, /*row_length=*/8,
                                            params.p, &l, &m, &err),
            retcode::SUCCESS) << err;
  params.l = l; params.m = m;

  core::Database db;
  ASSERT_EQ(db.SetupShape(/*num=*/64, /*row_length=*/8, params, &err),
            retcode::SUCCESS) << err;
  for (uint64_t i = 0; i < params.l; ++i) {
    for (uint64_t j = 0; j < params.m; ++j) {
      db.mutable_data().Set(i, j,
                             static_cast<uint32_t>((i * 7 + j) & 0xFF));
    }
  }
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));

  DoublePirHint src;
  ASSERT_EQ(HintGen::Compute(&db, params, &src, &err), retcode::SUCCESS) << err;

  std::string blob;
  ASSERT_EQ(SerializeHint(src, &blob, &err), retcode::SUCCESS) << err;

  DoublePirHint dst;
  ASSERT_EQ(DeserializeHint(blob, &dst, &err), retcode::SUCCESS) << err;
  ExpectHintsEqual(src, dst);
}

TEST(HintSerializeTest, EmptyHintRoundTrips) {
  // All-zero hint (default-constructed Matrix is 0x0). Still must
  // round-trip cleanly so callers can ship a "no work to do" sentinel.
  DoublePirHint src;
  std::string blob;
  std::string err;
  ASSERT_EQ(SerializeHint(src, &blob, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(blob.size(), 168u);  // No matrix payloads.

  DoublePirHint dst;
  ASSERT_EQ(DeserializeHint(blob, &dst, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(dst.A1.rows(), 0u);
  EXPECT_EQ(dst.A1.cols(), 0u);
  EXPECT_EQ(dst.H2_msg.rows(), 0u);
  EXPECT_EQ(dst.H2_msg.cols(), 0u);
  EXPECT_EQ(dst.info_after_setup.num, 0u);
}

}  // namespace
}  // namespace primihub::pir::double_pir
