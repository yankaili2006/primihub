/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * DoublePirProtocol unit tests — chunk 3 ships Init only. Setup /
 * Query / Answer / Recover have skeleton-returns-FAIL tests so the
 * harness is in place when chunks 4..6 fill them in. Mirrors the
 * SimplePIR protocol-test rhythm landed in 5a9e7616 / 47148373.
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir_protocol.h"

#include <random>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {
namespace {

// Builds a params + info pair sized like a tiny test database that
// satisfies DoublePIR's preconditions: params.l divisible by info.x,
// non-zero m / n / logq. Numbers picked to keep matrices small so
// the tests stay fast and the assertions easy to hand-verify.
struct InitFixture {
  core::LweParams params;
  core::DBinfo info;
  InitFixture() {
    params.m = 16;
    params.n = 4;
    params.l = 24;
    params.logq = 32;
    info.x = 3;  // 24 % 3 == 0
  }
};

TEST(DoublePirProtocolTest, InitShapesA1AndA2) {
  InitFixture f;
  core::Matrix A1, A2;
  std::string err;
  ASSERT_EQ(DoublePirProtocol::Init(f.params, f.info, &A1, &A2, &err),
            retcode::SUCCESS)
      << err;
  // A1 is params.m x params.n.
  EXPECT_EQ(A1.rows(), f.params.m);
  EXPECT_EQ(A1.cols(), f.params.n);
  // A2 is (params.l / info.x) x params.n.
  EXPECT_EQ(A2.rows(), f.params.l / f.info.x);
  EXPECT_EQ(A2.cols(), f.params.n);
}

TEST(DoublePirProtocolTest, InitFailsWhenLDoesNotDivideByX) {
  InitFixture f;
  f.info.x = 5;  // 24 % 5 != 0
  core::Matrix A1, A2;
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Init(f.params, f.info, &A1, &A2, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("must divide"), std::string::npos) << err;
}

TEST(DoublePirProtocolTest, InitFailsWhenParamsUnpopulated) {
  InitFixture f;
  f.params.m = 0;
  core::Matrix A1, A2;
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Init(f.params, f.info, &A1, &A2, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("populated"), std::string::npos) << err;
}

TEST(DoublePirProtocolTest, InitFailsOnNullOutputs) {
  InitFixture f;
  std::string err;
  core::Matrix A2;
  EXPECT_EQ(DoublePirProtocol::Init(f.params, f.info, nullptr, &A2, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("non-null"), std::string::npos) << err;
}

TEST(DoublePirProtocolTest, InitProducesUniformlyDistinctMatrices) {
  // Soft sanity check that UniformRandom actually filled the matrices
  // with non-zero data — catches accidental all-zero stubs and the
  // case where Init runs the wrong matrix into the wrong out param.
  InitFixture f;
  core::Matrix A1, A2;
  std::string err;
  ASSERT_EQ(DoublePirProtocol::Init(f.params, f.info, &A1, &A2, &err),
            retcode::SUCCESS);
  uint64_t a1_nonzero = 0;
  for (uint64_t i = 0; i < A1.size(); ++i) {
    if (A1.data()[i] != 0) ++a1_nonzero;
  }
  uint64_t a2_nonzero = 0;
  for (uint64_t i = 0; i < A2.size(); ++i) {
    if (A2.data()[i] != 0) ++a2_nonzero;
  }
  // With logq=32 (full uint32 range), the probability of any cell
  // being exactly 0 is ~2^-32; expecting >90% nonzero is conservative.
  EXPECT_GT(a1_nonzero, A1.size() * 9 / 10);
  EXPECT_GT(a2_nonzero, A2.size() * 9 / 10);
}

// ---- Skeleton-returns-FAIL coverage for Setup / Query / Answer / Recover.
// These tests pin the contract that subsequent chunks must satisfy: the
// FAIL message must point at the openspec task so callers know where
// to look. Lets us delete them in chunks 4..6 with a clear signal.

TEST(DoublePirProtocolSkeletonTest, RecoverReturnsFailPointingAtChunk6c) {
  core::Matrix A2, q1, H2, s1, a1;
  std::vector<core::Matrix> q2s, s2s, a2s;
  core::LweParams params;
  core::DBinfo info;
  uint64_t out = 0;
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Recover(0, A2, q1, q2s, H2, s1, s2s, a1, a2s,
                                        params, info, &out, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("chunk 6c"), std::string::npos) << err;
}


// ---- Setup (chunk 4b of DoublePIR port) ----
//
// Setup pulls in three Matrix kernel calls (Mul, Transpose, Mul
// again) plus pure-arithmetic Expand / ConcatCols / Squish /
// ScalarAdd / Concat. Real-mode (PIR_PIR_CORE_REAL set) is required
// for the kernel calls; stub-mode tests assert the FAIL path with
// a kernel-not-vendored message.

// Small helper: build a Database matching the InitFixture's shape so
// the protocol's preconditions on db->data() (L x M, info.x set,
// info.p / logq / squishing zero) and on A1 / A2 hold simultaneously.
class DoublePirSetupFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    params_.m = 16;
    params_.n = 4;
    params_.l = 24;
    params_.logq = 32;
    params_.p = 4;   // small power-of-2 p — keeps NumBasePDigits = 16
                      // and avoids LweParams::Pick complexity for unit
                      // tests; chunks 5/6 will exercise the real
                      // upstream params via integration tests.
    info_.x = 3;
    info_.p = params_.p;
    info_.logq = params_.logq;
    info_.basis = 0;       // not yet squished
    info_.squishing = 0;
    info_.cols = params_.m;
    db_.mutable_info() = info_;
    db_.mutable_data() = core::Matrix(params_.l, params_.m);
    // Fill with a deterministic non-zero pattern so we can assert
    // the post-Squish DB is non-trivially derived from it.
    for (uint64_t i = 0; i < params_.l; ++i) {
      for (uint64_t j = 0; j < params_.m; ++j) {
        db_.mutable_data().Set(i, j,
                               static_cast<uint32_t>((i * 7 + j) % params_.p));
      }
    }
    ASSERT_EQ(DoublePirProtocol::Init(params_, info_, &A1_, &A2_, &err_),
              retcode::SUCCESS) << err_;
  }
  core::LweParams params_;
  core::DBinfo info_;
  core::Database db_;
  core::Matrix A1_, A2_;
  std::string err_;
};

TEST_F(DoublePirSetupFixture, SetupRejectsNullOutputs) {
  core::Matrix H1, A2c, H2;
  EXPECT_EQ(
      DoublePirProtocol::Setup(&db_, A1_, A2_, params_, nullptr, &A2c, &H2,
                                &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("non-null"), std::string::npos) << err_;
}

TEST_F(DoublePirSetupFixture, SetupRejectsA1ShapeMismatch) {
  core::Matrix bad_A1(params_.m + 1, params_.n);
  core::Matrix H1, A2c, H2;
  EXPECT_EQ(
      DoublePirProtocol::Setup(&db_, bad_A1, A2_, params_, &H1, &A2c, &H2,
                                &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("A1 shape mismatch"), std::string::npos) << err_;
}

TEST_F(DoublePirSetupFixture, SetupRejectsA2ShapeMismatch) {
  core::Matrix bad_A2(params_.l / params_.n, params_.n);
  core::Matrix H1, A2c, H2;
  EXPECT_EQ(
      DoublePirProtocol::Setup(&db_, A1_, bad_A2, params_, &H1, &A2c, &H2,
                                &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("A2 shape mismatch"), std::string::npos) << err_;
}

TEST_F(DoublePirSetupFixture, SetupFailsLoudlyInStubMode) {
  // Stub-mode (kPirCoreKernelsVendored == false) hits the kernel
  // bridge on the very first Mul call; the failure message must
  // surface from Matrix::Mul's "not vendored" path.
  if (core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "vendored mode succeeds — see SetupProducesExpectedShapes";
  }
  core::Matrix H1, A2c, H2;
  EXPECT_EQ(
      DoublePirProtocol::Setup(&db_, A1_, A2_, params_, &H1, &A2c, &H2,
                                &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("DB * A1"), std::string::npos) << err_;
}

TEST_F(DoublePirSetupFixture, SetupProducesExpectedShapes) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "Setup correctness piggybacks on Matrix::Mul";
  }
  core::Matrix H1, A2c, H2;
  ASSERT_EQ(
      DoublePirProtocol::Setup(&db_, A1_, A2_, params_, &H1, &A2c, &H2,
                                &err_),
      retcode::SUCCESS) << err_;

  // H2 has shape (N * delta * X) x N.
  const uint64_t delta = params_.NumBasePDigits();
  const uint64_t rows_NdX = params_.n * delta * info_.x;
  EXPECT_EQ(H2.rows(), rows_NdX);
  EXPECT_EQ(H2.cols(), params_.n);

  // H1_squished: rows still (N * delta * X), cols = ceil((L/X) / 3).
  const uint64_t lx = params_.l / info_.x;
  const uint64_t expected_h1_cols = (lx + 2) / 3;  // ceil(lx / 3)
  EXPECT_EQ(H1.rows(), rows_NdX);
  EXPECT_EQ(H1.cols(), expected_h1_cols);

  // A2_copy_transposed: original A2 is (L/X) x N. After padding rows
  // up to multiple of 3 then transposing: cols x rows = N x padded.
  const uint64_t padded = ((lx + 2) / 3) * 3;
  EXPECT_EQ(A2c.rows(), params_.n);
  EXPECT_EQ(A2c.cols(), padded);

  // DB is now squished — basis=10, squishing=3, cols = ceil(M / 3).
  EXPECT_EQ(db_.info().basis, 10u);
  EXPECT_EQ(db_.info().squishing, 3u);
  EXPECT_EQ(db_.data().cols(), (params_.m + 2) / 3);
  EXPECT_EQ(db_.data().rows(), params_.l);

  // Sanity: H2 must contain at least one non-zero entry (else the
  // matrix multiplies upstream got short-circuited).
  bool h2_has_nonzero = false;
  for (uint64_t i = 0; i < H2.size(); ++i) {
    if (H2.data()[i] != 0) { h2_has_nonzero = true; break; }
  }
  EXPECT_TRUE(h2_has_nonzero);
}


// ---- Query (chunk 5 of DoublePIR port) ----

class DoublePirQueryFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Same shape as the Setup fixture, plus the ne / squishing
    // fields Query reads. Setup populates squishing as a side
    // effect; we set it explicitly here so Query can run without
    // calling Setup first (keeps the test surface narrow).
    params_.m = 16;
    params_.n = 4;
    params_.l = 24;
    params_.logq = 32;
    params_.p = 4;
    params_.sigma = core::kGaussianSigma;
    info_.x = 3;
    info_.ne = 6;     // info.ne must be divisible by info.x; 6 / 3 = 2 pairs.
    info_.squishing = 3;
    info_.p = params_.p;
    info_.logq = params_.logq;
    ASSERT_EQ(DoublePirProtocol::Init(params_, info_, &A1_, &A2_, &err_),
              retcode::SUCCESS) << err_;
  }
  core::LweParams params_;
  core::DBinfo info_;
  core::Matrix A1_, A2_;
  std::string err_;
};

TEST_F(DoublePirQueryFixture, QueryRejectsNullOutputs) {
  core::Matrix s1, q1;
  std::vector<core::Matrix> s2s, q2s;
  std::mt19937_64 rng(1);
  EXPECT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_, &rng,
                                      nullptr, &q1, &s2s, &q2s, &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("non-null"), std::string::npos) << err_;
}

TEST_F(DoublePirQueryFixture, QueryRejectsNeNotDivisibleByX) {
  info_.ne = 7;  // 7 % 3 != 0
  core::Matrix s1, q1;
  std::vector<core::Matrix> s2s, q2s;
  std::mt19937_64 rng(1);
  EXPECT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_, &rng,
                                      &s1, &q1, &s2s, &q2s, &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("divisible by"), std::string::npos) << err_;
}

TEST_F(DoublePirQueryFixture, QueryRejectsZeroSquishing) {
  info_.squishing = 0;
  core::Matrix s1, q1;
  std::vector<core::Matrix> s2s, q2s;
  std::mt19937_64 rng(1);
  EXPECT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_, &rng,
                                      &s1, &q1, &s2s, &q2s, &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("squishing"), std::string::npos) << err_;
}

TEST_F(DoublePirQueryFixture, QueryFailsLoudlyInStubMode) {
  if (core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "vendored mode succeeds — see QueryProducesExpectedShapes";
  }
  core::Matrix s1, q1;
  std::vector<core::Matrix> s2s, q2s;
  std::mt19937_64 rng(1);
  EXPECT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_, &rng,
                                      &s1, &q1, &s2s, &q2s, &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("A1 * secret1"), std::string::npos) << err_;
}

TEST_F(DoublePirQueryFixture, QueryProducesExpectedShapes) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "Query correctness piggybacks on Matrix::Mul";
  }
  core::Matrix s1, q1;
  std::vector<core::Matrix> s2s, q2s;
  std::mt19937_64 rng(1);
  ASSERT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_, &rng,
                                      &s1, &q1, &s2s, &q2s, &err_),
            retcode::SUCCESS) << err_;
  // secret1: N x 1.
  EXPECT_EQ(s1.rows(), params_.n);
  EXPECT_EQ(s1.cols(), 1u);
  // query1: M padded to multiple of squishing, x 1.
  const uint64_t q1_rows_expected =
      params_.m + ((params_.m % info_.squishing == 0)
                       ? 0
                       : info_.squishing - (params_.m % info_.squishing));
  EXPECT_EQ(q1.rows(), q1_rows_expected);
  EXPECT_EQ(q1.cols(), 1u);
  // secrets2 / queries2: exactly info.ne / info.x entries.
  const uint64_t num_pairs = info_.ne / info_.x;
  ASSERT_EQ(s2s.size(), num_pairs);
  ASSERT_EQ(q2s.size(), num_pairs);
  const uint64_t lx = params_.l / info_.x;
  const uint64_t q2_rows_expected =
      lx + ((lx % info_.squishing == 0)
                ? 0
                : info_.squishing - (lx % info_.squishing));
  for (uint64_t j = 0; j < num_pairs; ++j) {
    EXPECT_EQ(s2s[j].rows(), params_.n);
    EXPECT_EQ(s2s[j].cols(), 1u);
    EXPECT_EQ(q2s[j].rows(), q2_rows_expected);
    EXPECT_EQ(q2s[j].cols(), 1u);
  }
  // Sanity: query1 must contain at least one non-zero entry (the
  // Delta bump and Gaussian noise both contribute, A1 * secret1 is
  // uniformly random in Z_q, so all-zero is astronomically unlikely).
  bool q1_nonzero = false;
  for (uint64_t i = 0; i < q1.size(); ++i) {
    if (q1.data()[i] != 0) { q1_nonzero = true; break; }
  }
  EXPECT_TRUE(q1_nonzero);
}

// NOTE: a "QueryEncodesIndexAtCorrectPosition" test was attempted
// here but removed — it would have required deterministic
// reproducibility of the secret1 / secret2 random draws, which
// Matrix::UniformRandom does NOT provide (it seeds from
// std::random_device). End-to-end correctness of the index encoding
// is covered by the full Query-Answer-Recover round-trip test that
// lands in chunk 7 (DoublePirOperator wiring), where the recovered
// cell value at a specific index proves the bump position is right.



// ---- Answer (chunk 6b of DoublePIR port) ----

class DoublePirAnswerFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Larger m than the Setup fixture so the squished_db, query1
    // padding, and a1 reshaping all exercise non-trivial sizes.
    // params.n = 8 keeps A2_copy_transposed bRows=8 (a multiple of
    // 8, in case the kernel takes the short-rows branch — but in
    // practice for these shapes aRows > aCols and it takes the
    // long-rows branch).
    params_.m = 24;
    params_.n = 8;
    params_.l = 24;
    params_.logq = 32;
    params_.p = 4;
    params_.sigma = core::kGaussianSigma;
    info_.x = 3;
    info_.ne = 6;          // per_col = ne / x = 2.
    info_.squishing = 0;   // Setup populates basis/squishing (10/3).
    info_.basis = 0;
    info_.p = params_.p;
    info_.logq = params_.logq;
    info_.cols = params_.m;
    info_.row_length = 0;
    db_.mutable_info() = info_;
    db_.mutable_data() = core::Matrix(params_.l, params_.m);
    for (uint64_t i = 0; i < params_.l; ++i) {
      for (uint64_t j = 0; j < params_.m; ++j) {
        db_.mutable_data().Set(
            i, j, static_cast<uint32_t>((i * 7 + j) % params_.p));
      }
    }
    ASSERT_EQ(DoublePirProtocol::Init(params_, info_, &A1_, &A2_, &err_),
              retcode::SUCCESS) << err_;
  }
  core::LweParams params_;
  core::DBinfo info_;
  core::Database db_;
  core::Matrix A1_, A2_;
  std::string err_;
};

TEST_F(DoublePirAnswerFixture, AnswerRejectsNullOutputs) {
  core::Database squished_db = db_;
  squished_db.mutable_info().basis = 10;
  squished_db.mutable_info().squishing = 3;
  core::Matrix H1_squished, A2c, query1;
  std::vector<core::Matrix> queries2;
  EXPECT_EQ(
      DoublePirProtocol::Answer(squished_db, H1_squished, A2c, query1,
                                queries2, params_, nullptr, nullptr, &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("non-null"), std::string::npos) << err_;
}

TEST_F(DoublePirAnswerFixture, AnswerRejectsUnsquishedDb) {
  // DB still has basis=0 / squishing=0 (Setup hasn't run on it).
  core::Matrix H1_squished, A2c, query1, a1_out;
  std::vector<core::Matrix> queries2, a2s_out;
  EXPECT_EQ(
      DoublePirProtocol::Answer(db_, H1_squished, A2c, query1, queries2,
                                params_, &a1_out, &a2s_out, &err_),
      retcode::FAIL);
  EXPECT_NE(err_.find("basis=10"), std::string::npos) << err_;
}

TEST_F(DoublePirAnswerFixture, AnswerRejectsWrongQueries2Count) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs Setup to produce a squished DB";
  }
  core::Matrix H1, A2c, H2;
  ASSERT_EQ(
      DoublePirProtocol::Setup(&db_, A1_, A2_, params_, &H1, &A2c, &H2,
                                &err_),
      retcode::SUCCESS) << err_;
  // db_ is now squished; query1 needs the post-Setup squishing for
  // shape, but the precondition we want to trip fires BEFORE shape
  // checks — just pass empty queries2.
  core::Matrix query1(params_.m, 1);
  std::vector<core::Matrix> queries2;
  core::Matrix a1_out;
  std::vector<core::Matrix> a2s_out;
  EXPECT_EQ(DoublePirProtocol::Answer(db_, H1, A2c, query1, queries2, params_,
                                       &a1_out, &a2s_out, &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("queries2"), std::string::npos) << err_;
}

TEST_F(DoublePirAnswerFixture, AnswerFailsLoudlyInStubMode) {
  if (core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "vendored mode succeeds — see AnswerProducesExpectedShapes";
  }
  // Pretend the DB is squished so we get past the up-front guard;
  // the failure should then surface from MulVecPacked.
  core::Database squished_db = db_;
  squished_db.mutable_info().basis = 10;
  squished_db.mutable_info().squishing = 3;
  core::Matrix H1_squished, A2c, query1;
  std::vector<core::Matrix> queries2;
  queries2.resize(info_.ne / info_.x);  // right count, content irrelevant in stub mode
  core::Matrix a1_out;
  std::vector<core::Matrix> a2s_out;
  EXPECT_EQ(DoublePirProtocol::Answer(squished_db, H1_squished, A2c, query1,
                                       queries2, params_, &a1_out, &a2s_out,
                                       &err_),
            retcode::FAIL);
  EXPECT_NE(err_.find("squished_db * query1"), std::string::npos) << err_;
}

TEST_F(DoublePirAnswerFixture, AnswerProducesExpectedShapes) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "Answer kernel calls need vendored mode";
  }
  // Drive the protocol end-to-end up to Answer.
  core::Matrix H1_squished, A2c, H2;
  ASSERT_EQ(DoublePirProtocol::Setup(&db_, A1_, A2_, params_, &H1_squished,
                                       &A2c, &H2, &err_),
            retcode::SUCCESS) << err_;
  // Setup mutated db_.info() to populate basis=10 / squishing=3; pull
  // it back into the info we pass to Query so the AppendZeros padding
  // sees the right squishing factor.
  const core::DBinfo info_after_setup = db_.info();
  core::Matrix secret1, query1;
  std::vector<core::Matrix> secrets2, queries2;
  std::mt19937_64 rng(42);
  ASSERT_EQ(DoublePirProtocol::Query(0, A1_, A2_, params_, info_after_setup,
                                       &rng, &secret1, &query1, &secrets2,
                                       &queries2, &err_),
            retcode::SUCCESS) << err_;

  core::Matrix a1_out;
  std::vector<core::Matrix> a2s_out;
  ASSERT_EQ(DoublePirProtocol::Answer(db_, H1_squished, A2c, query1, queries2,
                                       params_, &a1_out, &a2s_out, &err_),
            retcode::SUCCESS) << err_;

  // a1 starts as (L, 1) — MulVecPacked output of squished_db * q1.
  // After Transpose+Expand+ConcatCols+Squish: a1_t.rows = delta * X,
  // a1_t.cols = ceil((L/X)/3). answer1 = MulTransposedPacked(a1_t,
  // A2_copy_transposed, 10, 3) — output shape (a1_t.rows, A2c.rows).
  // A2c.rows = N (A2c is the (padded A2).Transpose). So answer1 has
  // shape (delta * X, N).
  const uint64_t delta = params_.NumBasePDigits();
  const uint64_t rows_dX = delta * info_.x;
  EXPECT_EQ(a1_out.rows(), rows_dX);
  EXPECT_EQ(a1_out.cols(), params_.n);

  // answers2: interleaved [a2_0, h2_0, a2_1, h2_1, ...] — 2 * per_col entries.
  const uint64_t per_col = info_.ne / info_.x;
  ASSERT_EQ(a2s_out.size(), 2 * per_col);
  // a2 = MulVecPacked(H1_squished, q2) — shape (H1_squished.rows, 1)
  //      = (N * delta * X, 1).
  // h2 = MulVecPacked(a1_t, q2)         — shape (a1_t.rows, 1)
  //      = (delta * X, 1).
  for (uint64_t j = 0; j < per_col; ++j) {
    const core::Matrix& a2 = a2s_out[2 * j];
    const core::Matrix& h2 = a2s_out[2 * j + 1];
    EXPECT_EQ(a2.rows(), H1_squished.rows());
    EXPECT_EQ(a2.cols(), 1u);
    EXPECT_EQ(h2.rows(), rows_dX);
    EXPECT_EQ(h2.cols(), 1u);
  }
  // Sanity: at least one answers2 cell must be non-zero.
  bool nonzero_seen = false;
  for (const core::Matrix& m : a2s_out) {
    for (uint64_t i = 0; i < m.size(); ++i) {
      if (m.data()[i] != 0) { nonzero_seen = true; break; }
    }
    if (nonzero_seen) break;
  }
  EXPECT_TRUE(nonzero_seen);
}

}  // namespace
}  // namespace primihub::pir::double_pir
