/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirProtocol::Init / Setup tests. Init has no kernel dependency
 * (only Matrix::UniformRandom), so its shape + bound tests are
 * unconditional. Setup needs Matrix::Mul under PIR_PIR_CORE_REAL, so
 * its real-mode tests bifurcate on core::kPirCoreKernelsVendored.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_protocol.h"

namespace primihub::pir::simple_pir {
namespace {

// Helper: build a SimplePIR-ready LweParams + (l, m) from N entries
// of `row_length` bits each. Used by every test that exercises Setup.
core::LweParams MakeParams(uint64_t num_entries, uint64_t row_length,
                           std::string* err, retcode* rc) {
  core::LweParams params;
  params.n = 1024;
  params.logq = 32;
  // Pick a row that fits the eventual `m` value. ApproxSquareDB picks
  // a (l, m) close to sqrt(db_elems); for small `num_entries` the m
  // value comfortably fits the log_m=13 row.
  *rc = params.Pick(/*doublepir=*/false,
                    /*samples=*/static_cast<uint64_t>(1) << 13, err);
  if (*rc != retcode::SUCCESS) return params;
  uint64_t l = 0, m = 0;
  *rc = core::ApproxSquareDatabaseDims(num_entries, row_length, params.p,
                                       &l, &m, err);
  if (*rc != retcode::SUCCESS) return params;
  params.l = l;
  params.m = m;
  return params;
}

TEST(SimplePirInitTest, ProducesMatrixOfExpectedShape) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(1'000, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  core::Matrix A;
  ASSERT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::SUCCESS)
      << err;
  EXPECT_EQ(A.rows(), params.m);
  EXPECT_EQ(A.cols(), params.n);
  // Sanity: not all zeros (uniform random with millions of cells —
  // probability of all-zero is 0 to many decimal places).
  bool saw_nonzero = false;
  for (uint64_t i = 0; i < A.size(); ++i) {
    if (A.data()[i] != 0u) { saw_nonzero = true; break; }
  }
  EXPECT_TRUE(saw_nonzero);
}

TEST(SimplePirInitTest, FailsWhenParamsUninitialized) {
  core::LweParams params;
  // params.n, m, logq all zero
  core::Matrix A;
  std::string err;
  EXPECT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::FAIL);
  EXPECT_NE(err.find("params not initialized"), std::string::npos)
      << "must guide caller to call Pick/Approx first; got: " << err;
}

TEST(SimplePirInitTest, FailsWhenAOutNull) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(1'000, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  EXPECT_EQ(SimplePirProtocol::Init(params, /*A_out=*/nullptr, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("A_out is null"), std::string::npos) << err;
}

TEST(SimplePirSetupTest, BifurcatesOnVendoring) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(/*num=*/200, /*row_length=*/8,
                                      &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  core::Database db = core::Database::MakeRandom(200, 8, params, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  core::Matrix A;
  ASSERT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::SUCCESS)
      << err;

  core::Matrix H;
  auto setup_rc = SimplePirProtocol::Setup(&db, A, params, &H, &err);
  if (core::kPirCoreKernelsVendored) {
    ASSERT_EQ(setup_rc, retcode::SUCCESS)
        << "vendored mode: setup must succeed; err=" << err;
    EXPECT_EQ(H.rows(), params.l);
    EXPECT_EQ(H.cols(), params.n)
        << "H = DB(L x M) * A(M x N) -> L x N";
  } else {
    EXPECT_EQ(setup_rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_pir_core_real=1"), std::string::npos)
        << "stub error must guide callers to the activation flag; got: "
        << err;
  }
}

TEST(SimplePirSetupTest, FailsOnShapeMismatch) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  core::Database db = core::Database::MakeRandom(200, 8, params, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);

  // Wrong-shape A: pass an (m+1) x n matrix.
  core::Matrix bad_A = core::Matrix::Zeros(params.m + 1, params.n);
  core::Matrix H;
  EXPECT_EQ(SimplePirProtocol::Setup(&db, bad_A, params, &H, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("A shape mismatch"), std::string::npos)
      << "must surface shape mismatch up-front, not from kernel; got: "
      << err;
}

TEST(SimplePirSetupTest, FailsWhenParamsUninitialized) {
  core::LweParams params;
  core::Database db;
  core::Matrix A;
  core::Matrix H;
  std::string err;
  EXPECT_EQ(SimplePirProtocol::Setup(&db, A, params, &H, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("params not initialized"), std::string::npos) << err;
}

TEST(SimplePirSetupTest, ShiftDbByPHalf) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "shift assertion piggybacks on vendored Setup path";
  }
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);

  // Build a DB whose pre-Setup content is all zero so we can prove
  // the shift fired.
  core::Database db;
  ASSERT_EQ(db.SetupShape(200, 8, params, &err), retcode::SUCCESS) << err;
  // db.data is now Zeros(params.l, params.m) by SetupShape contract.

  core::Matrix A;
  ASSERT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::SUCCESS);

  core::Matrix H;
  ASSERT_EQ(SimplePirProtocol::Setup(&db, A, params, &H, &err),
            retcode::SUCCESS)
      << err;

  // After Setup, db.data should have been shifted by +p/2. Original
  // was all-zero; ScalarAdd(p/2) -> every cell is p/2.
  const uint32_t expected = static_cast<uint32_t>(params.p / 2);
  for (uint64_t i = 0; i < db.data().rows(); ++i) {
    for (uint64_t j = 0; j < db.data().cols(); ++j) {
      ASSERT_EQ(db.data().Get(i, j), expected)
          << "i=" << i << " j=" << j;
    }
  }
}

}  // namespace
}  // namespace primihub::pir::simple_pir
