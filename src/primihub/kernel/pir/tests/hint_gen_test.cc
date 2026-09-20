/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * HintGen unit tests — covers null guards, params-unpopulated guard,
 * the shape contract on the produced DoublePirHint, and per-stage
 * timing capture. End-to-end correctness of the hint is covered by
 * the operator EndToEnd test (which now drives HintGen internally).
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"

#include <gtest/gtest.h>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {
namespace {

TEST(HintGenTest, RejectsNullDb) {
  core::LweParams params;
  params.n = 4; params.logq = 32; params.p = 4;
  params.m = 8; params.l = 8;
  DoublePirHint hint;
  std::string err;
  EXPECT_EQ(HintGen::Compute(nullptr, params, &hint, &err), retcode::FAIL);
  EXPECT_NE(err.find("non-null"), std::string::npos) << err;
}

TEST(HintGenTest, RejectsNullHintOut) {
  core::Database db;
  core::LweParams params;
  params.n = 4; params.logq = 32; params.p = 4;
  params.m = 8; params.l = 8;
  std::string err;
  EXPECT_EQ(HintGen::Compute(&db, params, nullptr, &err), retcode::FAIL);
  EXPECT_NE(err.find("non-null"), std::string::npos) << err;
}

TEST(HintGenTest, RejectsUnpopulatedParams) {
  core::Database db;
  core::LweParams params;  // all zeros
  DoublePirHint hint;
  std::string err;
  EXPECT_EQ(HintGen::Compute(&db, params, &hint, &err), retcode::FAIL);
  EXPECT_NE(err.find("not fully populated"), std::string::npos) << err;
}

TEST(HintGenTest, ProducesExpectedShapesAndStats) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "HintGen Setup needs the kernel bridge";
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
  // Populate with arbitrary bytes — content doesn't affect the hint
  // shape, only the H2_msg values.
  for (uint64_t i = 0; i < params.l; ++i) {
    for (uint64_t j = 0; j < params.m; ++j) {
      db.mutable_data().Set(i, j,
                             static_cast<uint32_t>((i * 7 + j) & 0xFF));
    }
  }
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));

  DoublePirHint hint;
  HintGenStats stats;
  ASSERT_EQ(HintGen::Compute(&db, params, &hint, &err, &stats),
            retcode::SUCCESS) << err;

  // A1 shape: M x N
  EXPECT_EQ(hint.A1.rows(), params.m);
  EXPECT_EQ(hint.A1.cols(), params.n);
  // A2 shape: (L / x) x N
  EXPECT_EQ(hint.A2.rows(), params.l / hint.info_after_setup.x);
  EXPECT_EQ(hint.A2.cols(), params.n);

  // H2_msg shape: (N * delta * X) x N.
  const uint64_t delta = params.NumBasePDigits();
  const uint64_t rows_NdX =
      params.n * delta * hint.info_after_setup.x;
  EXPECT_EQ(hint.H2_msg.rows(), rows_NdX);
  EXPECT_EQ(hint.H2_msg.cols(), params.n);

  // info_after_setup must have basis=10 / squishing=3 populated.
  EXPECT_EQ(hint.info_after_setup.basis, 10u);
  EXPECT_EQ(hint.info_after_setup.squishing, 3u);

  // Timing must be non-negative and ~consistent with operator's
  // observed numbers (low ms range for this tiny DB).
  EXPECT_GE(stats.init_ms, 0.0);
  EXPECT_GE(stats.setup_ms, 0.0);
  EXPECT_LT(stats.init_ms, 5000.0);   // sanity ceiling — never minutes
  EXPECT_LT(stats.setup_ms, 5000.0);
}

TEST(HintGenTest, StatsOutOptional) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "HintGen Setup needs the kernel bridge";
  }
  core::LweParams params;
  params.n = 1024;
  params.logq = 32;
  std::string err;
  ASSERT_EQ(params.Pick(true, static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS);
  uint64_t l = 0, m = 0;
  ASSERT_EQ(core::ApproxSquareDatabaseDims(64, 8, params.p, &l, &m, &err),
            retcode::SUCCESS);
  params.l = l; params.m = m;
  core::Database db;
  ASSERT_EQ(db.SetupShape(64, 8, params, &err), retcode::SUCCESS);
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));
  DoublePirHint hint;
  // nullptr stats — must still succeed.
  ASSERT_EQ(HintGen::Compute(&db, params, &hint, &err, nullptr),
            retcode::SUCCESS) << err;
}

}  // namespace
}  // namespace primihub::pir::double_pir
