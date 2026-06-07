/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Database tests. Pure arithmetic + Matrix factories — no kernel
 * dependency, runs the same in both vendored and stub modes.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"

namespace primihub::pir::core {
namespace {

TEST(BasePTest, RoundtripsThroughReconstruct) {
  // Decompose 12345 in base 10: digits = [5, 4, 3, 2, 1].
  EXPECT_EQ(BaseP(10, 12345, 0), 5u);
  EXPECT_EQ(BaseP(10, 12345, 1), 4u);
  EXPECT_EQ(BaseP(10, 12345, 2), 3u);
  EXPECT_EQ(BaseP(10, 12345, 3), 2u);
  EXPECT_EQ(BaseP(10, 12345, 4), 1u);

  uint64_t digits[5] = {5, 4, 3, 2, 1};
  EXPECT_EQ(ReconstructFromBaseP(10, digits, 5), 12345u);
}

TEST(BasePTest, RoundtripsAtNonTrivialBase) {
  // Base-7 decomposition of 1000 — verify roundtrip without manually
  // computing digits.
  const uint64_t value = 1'000;
  const uint64_t base = 7;
  std::vector<uint64_t> digits;
  for (uint64_t i = 0; i < 8; ++i) {
    digits.push_back(BaseP(base, value, i));
  }
  EXPECT_EQ(ReconstructFromBaseP(base, digits.data(), digits.size()),
            value);
}

TEST(ComputeNumEntriesTest, MatchesUpstreamCeilFormula) {
  // p=991 (log2 ~ 9.95) for log_q=32 -> ceil(32 / 9.95) = 4.
  EXPECT_EQ(ComputeNumEntriesBaseP(991, 32), 4u);
  // p=2 (log2 = 1) for log_q=32 -> 32 entries.
  EXPECT_EQ(ComputeNumEntriesBaseP(2, 32), 32u);
}

TEST(NumDbEntriesTest, PacksWhenRowLengthLessThanLogP) {
  // p=991, log_p ~ 9.95, row_length=4 bits. log_p / row_length = 2
  // entries per Z_p elem (floor(9.95) / 4 = 2). For N=10 entries,
  // db_elems = ceil(10 / 2) = 5.
  uint64_t db_elems = 0, ne = 0, packing = 0;
  std::string err;
  ASSERT_EQ(
      NumDbEntries(/*n=*/10, /*row_length=*/4, /*p=*/991, &db_elems,
                   &ne, &packing, &err),
      retcode::SUCCESS)
      << err;
  EXPECT_EQ(db_elems, 5u);
  EXPECT_EQ(ne, 1u) << "packing path -> 1 Z_p elem holds multiple DB entries";
  EXPECT_EQ(packing, 2u);
}

TEST(NumDbEntriesTest, ExpandsWhenRowLengthExceedsLogP) {
  // p=991, log_p ~ 9.95, row_length=32 bits. Each DB entry needs
  // ceil(32 / 9.95) = 4 Z_p elems. For N=10 entries, db_elems =
  // 10 * 4 = 40.
  uint64_t db_elems = 0, ne = 0, packing = 0;
  std::string err;
  ASSERT_EQ(
      NumDbEntries(/*n=*/10, /*row_length=*/32, /*p=*/991, &db_elems,
                   &ne, &packing, &err),
      retcode::SUCCESS)
      << err;
  EXPECT_EQ(db_elems, 40u);
  EXPECT_EQ(ne, 4u);
  EXPECT_EQ(packing, 0u);
}

TEST(NumDbEntriesTest, FailsOnDegenerateInput) {
  uint64_t db_elems = 0, ne = 0, packing = 0;
  std::string err;
  EXPECT_EQ(
      NumDbEntries(/*n=*/0, /*row_length=*/8, /*p=*/991, &db_elems,
                   &ne, &packing, &err),
      retcode::FAIL);
}

TEST(ApproxSquareDatabaseDimsTest, ProducesValidLM) {
  uint64_t l = 0, m = 0;
  std::string err;
  ASSERT_EQ(
      ApproxSquareDatabaseDims(/*n=*/1'000'000, /*row_length=*/8,
                                /*p=*/991, &l, &m, &err),
      retcode::SUCCESS)
      << err;
  // Output invariants from the formula: l * m >= db_elems and l ~ sqrt
  // rounded up.
  EXPECT_GT(l, 0u);
  EXPECT_GT(m, 0u);
  EXPECT_GE(l * m, 125'000u)
      << "l*m must accommodate the packed elems (1M / 9 packing ~ "
         "112K)";
}

TEST(DatabaseSetupShapeTest, PopulatesDBinfoConsistently) {
  // Pick a valid LweParams row (log_m=13 fits 2^13 samples = 8192).
  LweParams params;
  params.n = 1024;
  params.logq = 32;
  std::string err;
  ASSERT_EQ(params.Pick(/*doublepir=*/false,
                        /*samples=*/static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS)
      << err;

  // Compute l/m to match the params. We need both to be set for
  // SetupShape's capacity check.
  uint64_t l = 0, m = 0;
  ASSERT_EQ(ApproxSquareDatabaseDims(/*n=*/2'000, /*row_length=*/8,
                                      /*p=*/params.p, &l, &m, &err),
            retcode::SUCCESS)
      << err;
  params.l = l;
  params.m = m;

  Database db;
  ASSERT_EQ(db.SetupShape(/*num=*/2'000, /*row_length=*/8, params, &err),
            retcode::SUCCESS)
      << err;
  EXPECT_EQ(db.info().num, 2'000u);
  EXPECT_EQ(db.info().row_length, 8u);
  EXPECT_EQ(db.info().p, params.p);
  EXPECT_EQ(db.info().logq, 32u);
  EXPECT_EQ(db.data().rows(), l);
  EXPECT_EQ(db.data().cols(), m);
  // Packing path: row_length=8 < log_p so packing > 0, ne = 1.
  EXPECT_EQ(db.info().ne, 1u);
  EXPECT_GT(db.info().packing, 0u);
}

TEST(DatabaseSetupShapeTest, FailsWhenParamsUninitialized) {
  LweParams params;
  // Don't Pick — p and logq stay 0.
  Database db;
  std::string err;
  EXPECT_EQ(db.SetupShape(100, 8, params, &err), retcode::FAIL);
  EXPECT_NE(err.find("LweParams not initialized"), std::string::npos)
      << "must guide caller to call Pick first; got: " << err;
}

TEST(DatabaseMakeRandomTest, ProducesValuesInShiftedRange) {
  LweParams params;
  params.n = 1024;
  params.logq = 32;
  std::string err;
  ASSERT_EQ(params.Pick(/*doublepir=*/false,
                        /*samples=*/static_cast<uint64_t>(1) << 13, &err),
            retcode::SUCCESS)
      << err;
  uint64_t l = 0, m = 0;
  ASSERT_EQ(ApproxSquareDatabaseDims(/*n=*/1'000, /*row_length=*/8,
                                      /*p=*/params.p, &l, &m, &err),
            retcode::SUCCESS)
      << err;
  params.l = l;
  params.m = m;

  retcode rc = retcode::FAIL;
  Database db = Database::MakeRandom(1'000, 8, params, &err, &rc);
  ASSERT_EQ(rc, retcode::SUCCESS) << err;
  EXPECT_EQ(db.data().rows(), l);
  EXPECT_EQ(db.data().cols(), m);
  // After ReduceMod(p) + ScalarSub(p/2), values land in
  // approximately [-p/2, p/2). With uint32 wrap-around, "negative"
  // values wrap to the upper half of uint32. Just assert that we
  // observe at least one wrap-around and at least one small value —
  // i.e. the distribution covers both halves.
  bool saw_low = false, saw_high = false;
  for (uint64_t i = 0; i < db.data().rows(); ++i) {
    for (uint64_t j = 0; j < db.data().cols(); ++j) {
      const uint32_t v = db.data().Get(i, j);
      if (v < params.p / 2) saw_low = true;
      if (v > (UINT32_MAX - params.p)) saw_high = true;
      if (saw_low && saw_high) break;
    }
    if (saw_low && saw_high) break;
  }
  EXPECT_TRUE(saw_low) << "expected at least one small uint32 value";
  EXPECT_TRUE(saw_high)
      << "expected at least one wrapped-negative uint32 value";
}

// --------------------------------------------------------------------
// Squish / Unsquish
// --------------------------------------------------------------------

// Builds a small, deterministic Database whose data fits in `basis_bits`
// for byte-level checks. Uses mutable_info / mutable_data so we do not
// have to go through SetupShape (which is heavier and unrelated here).
Database MakeSquishableDatabase(uint64_t basis_bits) {
  Database db;
  db.mutable_info().p = (uint64_t{1} << (basis_bits - 2));  // p = 2^(b-2)
  db.mutable_info().logq = 32;
  Matrix data(4, 6);
  const uint32_t mask = static_cast<uint32_t>((1ULL << basis_bits) - 1);
  for (uint64_t i = 0; i < 4; ++i) {
    for (uint64_t j = 0; j < 6; ++j) {
      data.Set(i, j, static_cast<uint32_t>((i * 6 + j + 1) & mask));
    }
  }
  db.mutable_data() = std::move(data);
  return db;
}

TEST(DatabaseSquishTest, SquishUpdatesDBinfoAndShape) {
  Database db = MakeSquishableDatabase(/*basis_bits=*/10);
  const uint64_t orig_cols = db.data().cols();
  std::string err;
  ASSERT_EQ(db.Squish(/*basis=*/10, /*squishing=*/3, &err), retcode::SUCCESS)
      << err;
  EXPECT_EQ(db.info().basis, 10u);
  EXPECT_EQ(db.info().squishing, 3u);
  EXPECT_EQ(db.info().cols, orig_cols);
  EXPECT_EQ(db.data().cols(), 2u);  // ceil(6/3) = 2
  EXPECT_EQ(db.data().rows(), 4u);
}

TEST(DatabaseSquishTest, SquishUnsquishRoundtrip) {
  Database db = MakeSquishableDatabase(/*basis_bits=*/10);
  Matrix original_data = db.data();
  std::string err;
  ASSERT_EQ(db.Squish(10, 3, &err), retcode::SUCCESS) << err;
  ASSERT_EQ(db.Unsquish(&err), retcode::SUCCESS) << err;

  EXPECT_EQ(db.info().basis, 0u);
  EXPECT_EQ(db.info().squishing, 0u);
  EXPECT_EQ(db.info().cols, 0u);
  ASSERT_EQ(db.data().rows(), original_data.rows());
  ASSERT_EQ(db.data().cols(), original_data.cols());
  for (uint64_t i = 0; i < db.data().rows(); ++i) {
    for (uint64_t j = 0; j < db.data().cols(); ++j) {
      EXPECT_EQ(db.data().Get(i, j), original_data.Get(i, j))
          << "roundtrip mismatch i=" << i << " j=" << j;
    }
  }
}

TEST(DatabaseSquishTest, FailsWhenAlreadySquished) {
  Database db = MakeSquishableDatabase(10);
  std::string err;
  ASSERT_EQ(db.Squish(10, 3, &err), retcode::SUCCESS);
  EXPECT_EQ(db.Squish(10, 3, &err), retcode::FAIL);
  EXPECT_NE(err.find("already squished"), std::string::npos)
      << "got: " << err;
}

TEST(DatabaseSquishTest, FailsWhenPExceedsBasisCapacity) {
  Database db = MakeSquishableDatabase(10);
  db.mutable_info().p = 32;  // > 2^4
  std::string err;
  EXPECT_EQ(db.Squish(/*basis=*/4, /*squishing=*/3, &err), retcode::FAIL);
  EXPECT_NE(err.find("p="), std::string::npos);
  EXPECT_NE(err.find("> 2^basis"), std::string::npos) << "got: " << err;
}

TEST(DatabaseSquishTest, FailsWhenLogqInsufficientForPacking) {
  Database db = MakeSquishableDatabase(10);
  db.mutable_info().logq = 20;  // 10 * 3 = 30 > 20
  std::string err;
  EXPECT_EQ(db.Squish(10, 3, &err), retcode::FAIL);
  EXPECT_NE(err.find("logq="), std::string::npos);
  EXPECT_NE(err.find("< basis*squishing="), std::string::npos)
      << "got: " << err;
}

TEST(DatabaseSquishTest, UnsquishFailsWhenNotSquished) {
  Database db = MakeSquishableDatabase(10);
  std::string err;
  EXPECT_EQ(db.Unsquish(&err), retcode::FAIL);
  EXPECT_NE(err.find("not squished"), std::string::npos) << "got: " << err;
}

}  // namespace
}  // namespace primihub::pir::core
