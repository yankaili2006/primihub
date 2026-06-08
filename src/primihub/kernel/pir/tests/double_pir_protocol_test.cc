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

TEST(DoublePirProtocolSkeletonTest, SetupReturnsFailPointingAtChunk4) {
  core::Database db;
  core::Matrix A1, A2, H1, A2c, H2;
  core::LweParams params;
  std::string err;
  EXPECT_EQ(
      DoublePirProtocol::Setup(&db, A1, A2, params, &H1, &A2c, &H2, &err),
      retcode::FAIL);
  EXPECT_NE(err.find("chunk 4"), std::string::npos) << err;
}

TEST(DoublePirProtocolSkeletonTest, QueryReturnsFailPointingAtChunk5) {
  core::Matrix A1, A2, s1, q1, s2, q2;
  core::LweParams params;
  core::DBinfo info;
  std::mt19937_64 rng(42);
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Query(0, A1, A2, params, info, &rng,
                                      &s1, &q1, &s2, &q2, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("chunk 5"), std::string::npos) << err;
}

TEST(DoublePirProtocolSkeletonTest, AnswerReturnsFailPointingAtChunk6) {
  core::Database db;
  core::Matrix H1, A2c, q1, q2, a1, a2;
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Answer(db, H1, A2c, q1, q2, &a1, &a2, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("chunk 6"), std::string::npos) << err;
}

TEST(DoublePirProtocolSkeletonTest, RecoverReturnsFailPointingAtChunk6) {
  core::Matrix q1, q2, H2, s1, s2, a1, a2;
  core::LweParams params;
  core::DBinfo info;
  uint64_t out = 0;
  std::string err;
  EXPECT_EQ(DoublePirProtocol::Recover(0, q1, q2, H2, s1, s2, a1, a2, params,
                                        info, &out, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("chunk 6"), std::string::npos) << err;
}

}  // namespace
}  // namespace primihub::pir::double_pir
