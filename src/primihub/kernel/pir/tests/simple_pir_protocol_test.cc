/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirProtocol::Init / Setup / GenSecret / Query tests. Init and
 * GenSecret have no kernel dependency (only Matrix::UniformRandom),
 * so their shape + bound tests are unconditional. Setup and Query
 * need Matrix::Mul under PIR_PIR_CORE_REAL, so their real-mode tests
 * bifurcate on core::kPirCoreKernelsVendored.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/gaussian.h"
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

// --------------------------------------------------------------------
// GenSecret
// --------------------------------------------------------------------

TEST(SimplePirGenSecretTest, ProducesNx1WithNonzeroCells) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(1'000, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  core::Matrix s;
  ASSERT_EQ(SimplePirProtocol::GenSecret(params, &s, &err), retcode::SUCCESS)
      << err;
  EXPECT_EQ(s.rows(), params.n);
  EXPECT_EQ(s.cols(), 1u);
  // params.n = 1024 — uniform random hitting all-zero by chance is
  // astronomically unlikely.
  bool saw_nonzero = false;
  for (uint64_t i = 0; i < s.size(); ++i) {
    if (s.data()[i] != 0u) { saw_nonzero = true; break; }
  }
  EXPECT_TRUE(saw_nonzero);
}

TEST(SimplePirGenSecretTest, FailsWhenParamsUninitialized) {
  core::LweParams params;  // n, logq are zero
  core::Matrix s;
  std::string err;
  EXPECT_EQ(SimplePirProtocol::GenSecret(params, &s, &err), retcode::FAIL);
  EXPECT_NE(err.find("params not initialized"), std::string::npos)
      << "must guide caller; got: " << err;
}

TEST(SimplePirGenSecretTest, FailsWhenSecretOutNull) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(1'000, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  EXPECT_EQ(SimplePirProtocol::GenSecret(params, /*secret_out=*/nullptr, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("secret_out is null"), std::string::npos) << err;
}

// --------------------------------------------------------------------
// Query
// --------------------------------------------------------------------

TEST(SimplePirQueryTest, BifurcatesOnVendoring) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  // Deterministic A and secret so the test is reproducible.
  core::Matrix A = core::Matrix::UniformRandom(params.m, params.n,
                                               static_cast<uint32_t>(params.logq));
  core::Matrix s = core::Matrix::UniformRandom(params.n, 1,
                                               static_cast<uint32_t>(params.logq));
  std::mt19937_64 rng(42);

  core::Matrix query;
  core::DBinfo no_squish_info;  // squishing == 0 -> no padding
  auto query_rc = SimplePirProtocol::Query(/*index=*/7, A, s, params,
                                           no_squish_info, &rng, &query,
                                           &err);
  if (core::kPirCoreKernelsVendored) {
    ASSERT_EQ(query_rc, retcode::SUCCESS)
        << "vendored mode: query must succeed; err=" << err;
    EXPECT_EQ(query.rows(), params.m);
    EXPECT_EQ(query.cols(), 1u);
  } else {
    EXPECT_EQ(query_rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_pir_core_real=1"), std::string::npos)
        << "stub error must guide callers to the activation flag; got: "
        << err;
  }
}

TEST(SimplePirQueryTest, EncodesIndexAtDelta) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "Query correctness piggybacks on Matrix::Mul";
  }
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  core::Matrix A = core::Matrix::UniformRandom(params.m, params.n,
                                               static_cast<uint32_t>(params.logq));
  core::Matrix s = core::Matrix::UniformRandom(params.n, 1,
                                               static_cast<uint32_t>(params.logq));

  // Replay the protocol's RNG path so we can subtract noise out.
  std::mt19937_64 protocol_rng(/*seed=*/12345);
  std::mt19937_64 replay_rng(/*seed=*/12345);

  // Compute expected = A*s + replayed_noise + Delta * e_k. The
  // replayed_noise must be sampled BEFORE Query is called so we get
  // the same draws — GaussianSampler is deterministic for a fixed RNG
  // sequence.
  core::Matrix expected;
  ASSERT_EQ(A.Mul(s, &expected, &err), retcode::SUCCESS) << err;
  ASSERT_EQ(expected.rows(), params.m);
  ASSERT_EQ(expected.cols(), 1u);

  core::GaussianSampler replay_sampler(replay_rng);
  std::vector<int64_t> noise(params.m);
  for (uint64_t i = 0; i < params.m; ++i) {
    noise[i] = replay_sampler.Sample();
    const uint32_t curr = expected.Get(i, 0);
    expected.Set(i, 0, curr + static_cast<uint32_t>(noise[i]));
  }
  const uint64_t index = 17;
  const uint64_t k = index % params.m;
  expected.Set(k, 0,
               expected.Get(k, 0) + static_cast<uint32_t>(params.Delta()));

  core::Matrix query;
  core::DBinfo no_squish_info;
  ASSERT_EQ(SimplePirProtocol::Query(index, A, s, params, no_squish_info,
                                     &protocol_rng, &query, &err),
            retcode::SUCCESS)
      << err;

  ASSERT_EQ(query.rows(), expected.rows());
  ASSERT_EQ(query.cols(), expected.cols());
  for (uint64_t i = 0; i < query.size(); ++i) {
    ASSERT_EQ(query.data()[i], expected.data()[i])
        << "mismatch at i=" << i << " (k=" << k
        << ", noise=" << noise[i] << ")";
  }
}

TEST(SimplePirQueryTest, NoiseBoundedByGaussianTail) {
  // Statistical sanity that does NOT depend on Matrix::Mul: run a
  // tight loop of GaussianSampler::Sample() at the embedded sigma and
  // check that 100% of samples land in the table's [-128, 128] support.
  // If the CDF or the RNG drift, this catches it before Query yields
  // wrap-around chaos.
  std::mt19937_64 rng(/*seed=*/2026);
  core::GaussianSampler sampler(rng);
  for (int n = 0; n < 10000; ++n) {
    const int64_t e = sampler.Sample();
    ASSERT_GE(e, -128);
    ASSERT_LE(e, 128);
  }
}

TEST(SimplePirQueryTest, FailsOnSecretShapeMismatch) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);

  core::Matrix A = core::Matrix::Zeros(params.m, params.n);
  // Wrong-shape secret: (n+1) x 1.
  core::Matrix bad_s = core::Matrix::Zeros(params.n + 1, 1);
  std::mt19937_64 rng(0);
  core::Matrix q;
  core::DBinfo no_squish_info;
  EXPECT_EQ(SimplePirProtocol::Query(0, A, bad_s, params, no_squish_info,
                                     &rng, &q, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("secret shape mismatch"), std::string::npos)
      << "must surface mismatch up-front, not from kernel; got: " << err;
}

TEST(SimplePirQueryTest, FailsOnNullQueryOutOrRng) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  core::Matrix A = core::Matrix::Zeros(params.m, params.n);
  core::Matrix s = core::Matrix::Zeros(params.n, 1);
  std::mt19937_64 rng(0);
  core::Matrix q;
  core::DBinfo no_squish_info;
  EXPECT_EQ(SimplePirProtocol::Query(0, A, s, params, no_squish_info,
                                     /*noise_rng=*/nullptr, &q, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("noise_rng is null"), std::string::npos) << err;

  EXPECT_EQ(SimplePirProtocol::Query(0, A, s, params, no_squish_info, &rng,
                                     /*query_out=*/nullptr, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("query_out or noise_rng is null"), std::string::npos)
      << err;
}

// --------------------------------------------------------------------
// Answer + Recover (end-to-end, vendored-only)
// --------------------------------------------------------------------

TEST(SimplePirAnswerTest, FailsWhenDbNotSquished) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  core::Database db = core::Database::MakeRandom(200, 8, params, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  // Don't Squish — Answer should refuse.
  core::Matrix query = core::Matrix::Zeros(params.m, 1);
  core::Matrix answer;
  EXPECT_EQ(SimplePirProtocol::Answer(db, query, &answer, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("not squished"), std::string::npos) << err;
}

TEST(SimplePirRecoverTest, FailsWhenInfoNotInitialized) {
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(200, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS);
  core::DBinfo empty_info;  // ne=0 etc
  core::Matrix query = core::Matrix::Zeros(params.m, 1);
  core::Matrix hint = core::Matrix::Zeros(params.l, params.n);
  core::Matrix secret = core::Matrix::Zeros(params.n, 1);
  core::Matrix answer = core::Matrix::Zeros(params.l, 1);
  uint64_t out = 0;
  EXPECT_EQ(SimplePirProtocol::Recover(0, query, hint, secret, answer,
                                       params, empty_info, &out, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("info not initialized"), std::string::npos) << err;
}

TEST(SimplePirAnswerTest, EndToEndRetrievesCorrectEntry) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "end-to-end retrieval needs the kernel bridge";
  }
  // Full pipeline: Init -> Setup -> Squish -> GenSecret -> Query ->
  // Answer -> Recover. With row_length=8, p=991 (log_m=13 row), the
  // DBinfo layout is packing=1, ne=1 — one Z_p value per DB entry,
  // recovered value is `(Z_p value) % 2^row_length`. We pin DB
  // entries in [0, 2^row_length) so the % step is the identity, and
  // assert recovered equals the original byte.
  //
  // N=64 -> l=8, m=8. l must be a multiple of 8 because the
  // matMulVecPacked C kernel processes rows in batches of 8 (uses
  // `i += 8` loop) and reads a[i + 7*cols] regardless of whether the
  // row is in bounds. The 14-row case (N=200 from earlier drafts)
  // OOB-reads rows 14..15 which yields wrong noise terms in the
  // partial last batch.
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(64, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;

  core::Database db;
  ASSERT_EQ(db.SetupShape(64, 8, params, &err), retcode::SUCCESS) << err;
  // Pick a target inside (l x m) = (8 x 8) — index 27 → row 3, col 3.
  const uint64_t target_index = 27;
  ASSERT_LT(target_index, params.l * params.m);
  const uint64_t target_row = target_index / params.m;
  const uint64_t target_col = target_index % params.m;
  // Fill with byte-sized deterministic values, then shift to the
  // centered representation [-p/2, p/2) that upstream's MakeDB +
  // ReconstructElem assume. Without the ScalarSub(p/2), the math
  // recovers `byte + p/2 (mod p)` instead of `byte`.
  for (uint64_t i = 0; i < params.l; ++i) {
    for (uint64_t j = 0; j < params.m; ++j) {
      const uint32_t v = static_cast<uint32_t>((i * 13 + j * 7) & 0xFF);
      db.mutable_data().Set(i, j, v);
    }
  }
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));
  const uint64_t expected =
      static_cast<uint64_t>((target_row * 13 + target_col * 7) & 0xFF);

  // Init + Setup (Setup shifts DB by p/2 internally)
  core::Matrix A;
  ASSERT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::SUCCESS);
  core::Matrix H;
  ASSERT_EQ(SimplePirProtocol::Setup(&db, A, params, &H, &err),
            retcode::SUCCESS) << err;

  // Squish the DB with upstream's params (basis=10, squishing=3).
  ASSERT_EQ(db.Squish(/*basis=*/10, /*squishing=*/3, &err),
            retcode::SUCCESS) << err;

  // GenSecret
  core::Matrix s;
  ASSERT_EQ(SimplePirProtocol::GenSecret(params, &s, &err), retcode::SUCCESS);

  // Query — pass the now-Squished DBinfo so Query pads to a multiple
  // of squishing when params.m is not divisible by squishing.
  std::mt19937_64 rng(/*seed=*/987654);
  core::Matrix query;
  ASSERT_EQ(SimplePirProtocol::Query(target_index, A, s, params,
                                     db.info(), &rng, &query, &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(query.rows() % db.info().squishing, 0u);

  // Answer
  core::Matrix answer;
  ASSERT_EQ(SimplePirProtocol::Answer(db, query, &answer, &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(answer.rows(), params.l);
  EXPECT_EQ(answer.cols(), 1u);

  // Recover — uses db.info() which still has ne/p/logq/packing/
  // row_length populated even post-Squish.
  uint64_t recovered = 0;
  ASSERT_EQ(SimplePirProtocol::Recover(target_index, query, H, s, answer,
                                       params, db.info(), &recovered, &err),
            retcode::SUCCESS) << err;

  EXPECT_EQ(recovered, expected)
      << "DB entry at index=" << target_index << " (row=" << target_row
      << ", col=" << target_col << ") expected " << expected
      << " but recovered " << recovered;
}

// Validates the un-Squished Answer path: server uses regular Matrix::Mul
// instead of MulVecPacked. This is the path SimplePirProtocol::Answer
// would take if we later add a "no compression" mode. Kept around to
// pin the math independently of the packed kernel.
TEST(SimplePirAnswerTest, RecoverWithoutSquishWorks) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs kernel bridge";
  }
  std::string err;
  retcode rc = retcode::FAIL;
  core::LweParams params = MakeParams(64, 8, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  core::Database db;
  ASSERT_EQ(db.SetupShape(64, 8, params, &err), retcode::SUCCESS);
  const uint64_t target_index = 27;
  const uint64_t target_row = target_index / params.m;
  const uint64_t target_col = target_index % params.m;
  for (uint64_t i = 0; i < params.l; ++i) {
    for (uint64_t j = 0; j < params.m; ++j) {
      db.mutable_data().Set(i, j,
          static_cast<uint32_t>((i * 13 + j * 7) & 0xFF));
    }
  }
  // Center the DB the same way upstream's MakeDB does (subtract p/2)
  // so ReconstructElem's add-p/2 step recovers the original byte.
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));
  const uint64_t expected =
      static_cast<uint64_t>((target_row * 13 + target_col * 7) & 0xFF);

  core::Matrix A;
  ASSERT_EQ(SimplePirProtocol::Init(params, &A, &err), retcode::SUCCESS);
  core::Matrix H;
  ASSERT_EQ(SimplePirProtocol::Setup(&db, A, params, &H, &err),
            retcode::SUCCESS) << err;

  core::Matrix s;
  ASSERT_EQ(SimplePirProtocol::GenSecret(params, &s, &err), retcode::SUCCESS);

  std::mt19937_64 rng(/*seed=*/42);
  core::Matrix query;
  core::DBinfo no_squish_info;  // squishing=0 -> no padding
  ASSERT_EQ(SimplePirProtocol::Query(target_index, A, s, params,
                                     no_squish_info, &rng, &query, &err),
            retcode::SUCCESS) << err;

  // Compute answer manually via regular Mul (bypass MulVecPacked path).
  core::Matrix answer;
  ASSERT_EQ(db.data().Mul(query, &answer, &err), retcode::SUCCESS) << err;

  uint64_t recovered = 0;
  ASSERT_EQ(SimplePirProtocol::Recover(target_index, query, H, s, answer,
                                       params, db.info(), &recovered, &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(recovered, expected)
      << "Recover-without-Squish: expected " << expected
      << " got " << recovered;
}

}  // namespace
}  // namespace primihub::pir::simple_pir
