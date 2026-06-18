/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_params_test — chunk 4 verification of BaseParams + Common
 * Params. Strategy:
 *   * Use BaseParams::NewWithSeed for deterministic test fixtures.
 *   * Cross-check GenerateParamsRhs by re-deriving the LWE matrix
 *     and dot-product math by hand for tiny dim/m parameters.
 *   * Verify CommonParams::FromBaseParams returns the matrix the
 *     base params was seeded from.
 *   * For CommonParams::MultLeft (non-deterministic due to
 *     RandomTernary noise), assert the noise is in {0, 1, -1}
 *     mod 2^32 relative to the deterministic vec_mult.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base64.h"  // NOLINT — for Database fixture encoding
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

namespace primihub::pir::frodo {
namespace {

SeedBytes IotaSeed(std::uint8_t start) {
  SeedBytes s;
  for (std::size_t i = 0; i < 32; ++i) {
    s[i] = static_cast<std::uint8_t>(start + i);
  }
  return s;
}

// Builds a 2x1 Database via the data ctor (skipping base64 round-
// trip for simplicity). Column-form entries = {{10, 20}}. The DB
// has m=2 rows, elem_size=8, plaintext_bits=5 (row_width=2,
// remainder=3 — clean roundtrip per chunk 3c lessons).
Database MakeTinyDb() {
  std::vector<std::vector<std::uint32_t>> entries = {
      {10u, 20u},  // column 0
      {30u, 40u},  // column 1
  };
  return Database(std::move(entries), /*m=*/2, /*elem_size=*/8,
                  /*plaintext_bits=*/5);
}

// ---- BaseParams ------------------------------------------------

TEST(FrodoBaseParamsTest, NewWithSeed_FillsAllFields) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(0);
  const std::size_t dim = 3;
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, dim, seed, &params, &err),
            retcode::SUCCESS)
      << err;
  EXPECT_EQ(params.GetDim(), dim);
  EXPECT_EQ(params.GetTotalRecords(), 2u);
  EXPECT_EQ(params.GetElemSize(), 8u);
  EXPECT_EQ(params.GetPlaintextBits(), 5u);
  EXPECT_EQ(params.GetPublicSeed(), seed);
  // rhs shape: w x dim = db.GetMatrixWidthSelf() x dim.
  // db row_width = ceil(8 / 5) = 2; dim = 3.
  ASSERT_EQ(params.RhsForTest().size(), 2u);
  for (const auto& col : params.RhsForTest()) {
    EXPECT_EQ(col.size(), dim);
  }
}

TEST(FrodoBaseParamsTest, NewWithSeed_Deterministic_SameSeedSameRhs) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(42);
  BaseParams a, b;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, 4, seed, &a, &err),
            retcode::SUCCESS);
  ASSERT_EQ(BaseParams::NewWithSeed(db, 4, seed, &b, &err),
            retcode::SUCCESS);
  EXPECT_EQ(a.RhsForTest(), b.RhsForTest())
      << "same seed must yield identical rhs — required for "
      << "client/server matrix A agreement";
  EXPECT_EQ(a.GetPublicSeed(), b.GetPublicSeed());
}

TEST(FrodoBaseParamsTest, New_ProducesFreshSeedAcrossCalls) {
  // GenerateSeed is OS-RNG-backed so two BaseParams::New calls
  // must produce different public_seeds. Collision probability
  // is 2^-256.
  const auto db = MakeTinyDb();
  BaseParams a, b;
  std::string err;
  ASSERT_EQ(BaseParams::New(db, 4, &a, &err), retcode::SUCCESS);
  ASSERT_EQ(BaseParams::New(db, 4, &b, &err), retcode::SUCCESS);
  EXPECT_NE(a.GetPublicSeed(), b.GetPublicSeed())
      << "two BaseParams::New calls returned the same seed; "
      << "OS RNG is wedged";
  EXPECT_NE(a.RhsForTest(), b.RhsForTest())
      << "two BaseParams::New calls produced identical rhs — "
      << "extremely unlikely";
}

TEST(FrodoBaseParamsTest, GenerateParamsRhs_MatchesHandComputed) {
  // Hand-trace for tiny db + dim. Verifies rhs[i][j] =
  // db.VecMult(lhs_swapped[j], i) where lhs_swapped = transpose
  // of GenerateLweMatrixFromSeed(seed, dim=2, m=2).
  const auto db = MakeTinyDb();  // entries (column-form) = {{10,20}, {30,40}}
  const auto seed = IotaSeed(7);
  const std::size_t dim = 2;
  const std::size_t m = db.GetMatrixHeight();  // 2

  // Reference: derive A independently.
  const auto a_matrix = GenerateLweMatrixFromSeed(seed, dim, m);
  const auto a_swapped = SwapMatrixFmt(a_matrix);
  // a_swapped has dim=2 columns of m=2 u32s each.

  const auto rhs = BaseParams::GenerateParamsRhs(db, seed, dim, m);
  ASSERT_EQ(rhs.size(), db.GetMatrixWidthSelf());  // = 2

  for (std::size_t i = 0; i < rhs.size(); ++i) {
    ASSERT_EQ(rhs[i].size(), dim);
    for (std::size_t j = 0; j < dim; ++j) {
      // Expected: dot product of a_swapped[j] (m u32s) against
      // db's i-th column (m u32s) with u32 wrapping arithmetic.
      const auto& r = a_swapped[j];
      // chunk g-4: EntriesForTest() now returns by value (materialised
      // on demand from the ColMajorMatrix backing). Bind to a local
      // copy so the temporary outlives the inner loop.
      const auto dbcol = db.EntriesForTest()[i];
      std::uint32_t expected = 0;
      for (std::size_t k = 0; k < m; ++k) {
        expected += r[k] * dbcol[k];  // wraps mod 2^32 by spec
      }
      EXPECT_EQ(rhs[i][j], expected)
          << "rhs[" << i << "][" << j << "] mismatch";
    }
  }
}

TEST(FrodoBaseParamsTest, MultRight_HandComputed) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(13);
  const std::size_t dim = 3;
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, dim, seed, &params, &err),
            retcode::SUCCESS);

  const std::vector<std::uint32_t> s = {2u, 5u, 7u};  // length dim
  std::vector<std::uint32_t> got;
  ASSERT_EQ(params.MultRight(s, &got, &err), retcode::SUCCESS) << err;

  // Expected: for each rhs column i, sum_j s[j] * rhs[i][j].
  const auto& rhs = params.RhsForTest();
  ASSERT_EQ(got.size(), rhs.size());
  for (std::size_t i = 0; i < rhs.size(); ++i) {
    std::uint32_t expected = 0;
    ASSERT_EQ(s.size(), rhs[i].size());
    for (std::size_t j = 0; j < s.size(); ++j) {
      expected += s[j] * rhs[i][j];
    }
    EXPECT_EQ(got[i], expected) << "rhs column " << i;
  }
}

TEST(FrodoBaseParamsTest, MultRight_SizeMismatch_Fails) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(99);
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, 3, seed, &params, &err),
            retcode::SUCCESS);
  // s.size() != dim, so the vec_mult against each rhs column fails.
  const std::vector<std::uint32_t> bad_s = {1u, 2u};
  std::vector<std::uint32_t> got;
  EXPECT_EQ(params.MultRight(bad_s, &got, &err), retcode::FAIL);
  EXPECT_NE(err.find("rhs column"), std::string::npos) << err;
  EXPECT_NE(err.find("ErrorUnexpectedInputSize"), std::string::npos)
      << err;
}

// ---- CommonParams ----------------------------------------------

TEST(FrodoCommonParamsTest, FromBaseParams_ReturnsLwe) {
  // Reference: independent direct call to GenerateLweMatrixFromSeed.
  // Then CommonParams::FromBaseParams must give the same matrix.
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(11);
  const std::size_t dim = 4;
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, dim, seed, &params, &err),
            retcode::SUCCESS);

  const auto expected =
      GenerateLweMatrixFromSeed(seed, dim, db.GetMatrixHeight());
  const auto common = CommonParams::FromBaseParams(params);
  EXPECT_EQ(common.AsMatrix(), expected)
      << "CommonParams::FromBaseParams must re-derive the LWE "
      << "matrix from the public seed";
}

TEST(FrodoCommonParamsTest, MultLeft_NoiseIsTernary) {
  // For each column i: out[i] = vec_mult(s, cols[i]) + e where
  // e ∈ {0, 1, -1} mod 2^32 = {0, 1, 0xFFFFFFFF}. Compute
  // diff = out[i] - vec_mult(s, cols[i]) (mod 2^32) and check
  // it lands in that set.
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(21);
  const std::size_t dim = 3;
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, dim, seed, &params, &err),
            retcode::SUCCESS);
  const auto common = CommonParams::FromBaseParams(params);

  // s vector length must match column size = dim.
  const std::vector<std::uint32_t> s = {3u, 11u, 17u};
  std::vector<std::uint32_t> got;
  ASSERT_EQ(common.MultLeft(s, &got, &err), retcode::SUCCESS) << err;
  ASSERT_EQ(got.size(), common.AsMatrix().size());

  for (std::size_t i = 0; i < got.size(); ++i) {
    std::uint32_t s_a = 0;
    ASSERT_EQ(VecMultU32U32(s, common.AsMatrix()[i], &s_a, &err),
              retcode::SUCCESS) << err;
    const std::uint32_t diff = got[i] - s_a;  // u32 wraps
    const bool ok = (diff == 0u) || (diff == 1u) ||
                    (diff == 0xFFFFFFFFu);
    EXPECT_TRUE(ok) << "noise at column " << i << " was " << diff
                    << " — not in {0, 1, 0xFFFFFFFF}";
  }
}

TEST(FrodoCommonParamsTest, MultLeft_SizeMismatch_Fails) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(31);
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, 3, seed, &params, &err),
            retcode::SUCCESS);
  const auto common = CommonParams::FromBaseParams(params);

  const std::vector<std::uint32_t> bad_s = {1u};  // size != dim
  std::vector<std::uint32_t> got;
  EXPECT_EQ(common.MultLeft(bad_s, &got, &err), retcode::FAIL);
  EXPECT_NE(err.find("matrix column"), std::string::npos) << err;
}

TEST(FrodoCommonParamsTest, MultLeft_NullOut_Fails) {
  const auto db = MakeTinyDb();
  const auto seed = IotaSeed(41);
  BaseParams params;
  std::string err;
  ASSERT_EQ(BaseParams::NewWithSeed(db, 2, seed, &params, &err),
            retcode::SUCCESS);
  const auto common = CommonParams::FromBaseParams(params);
  EXPECT_EQ(common.MultLeft({1u, 2u}, nullptr, &err), retcode::FAIL);
  EXPECT_NE(err.find("out must be non-null"), std::string::npos);
}

TEST(FrodoCommonParamsTest, AsMatrix_ConstRefStable) {
  // AsMatrix returns const&; the underlying object must stay
  // alive through the matrix's lifetime.
  std::vector<std::vector<std::uint32_t>> seed_mat = {
      {0xDEADBEEFu, 0xCAFEBABEu},
      {0x12345678u, 0x87654321u},
  };
  const CommonParams common(std::move(seed_mat));
  EXPECT_EQ(common.AsMatrix().size(), 2u);
  EXPECT_EQ(common.AsMatrix()[0][0], 0xDEADBEEFu);
  EXPECT_EQ(common.AsMatrix()[1][1], 0x87654321u);
}

}  // namespace
}  // namespace primihub::pir::frodo
