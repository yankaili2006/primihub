/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_database_test — verifies the Database struct port. Chunk
 * 3a covers the data ctor + simple methods; chunk 3b adds
 * ConstructRows + Database::New from base64-encoded strings.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base64.h"  // NOLINT — for fixture encoding

namespace primihub::pir::frodo {
namespace {

// ---- Chunk 3a tests (struct + simple methods) ------------------

// Helper: a 3-row, 2-col DB in column-form (entries[col][row]).
//   row 0: {1, 2}
//   row 1: {3, 4}
//   row 2: {5, 6}
// In column-form layout: entries = {{1,3,5}, {2,4,6}}.
Database MakeSmallDb() {
  std::vector<std::vector<std::uint32_t>> entries = {
      {1u, 3u, 5u},
      {2u, 4u, 6u},
  };
  return Database(std::move(entries), /*m=*/3, /*elem_size=*/16,
                  /*plaintext_bits=*/8);
}

TEST(FrodoDatabaseTest, Getters_ReturnConstructorArgs) {
  const auto db = MakeSmallDb();
  EXPECT_EQ(db.GetMatrixHeight(), 3u);
  EXPECT_EQ(db.GetElemSize(), 16u);
  EXPECT_EQ(db.GetPlaintextBits(), 8u);
  // GetMatrixWidthSelf = ceil(16 / 8) = 2.
  EXPECT_EQ(db.GetMatrixWidthSelf(), 2u);
}

TEST(FrodoDatabaseTest, GetMatrixWidth_Static_ExactDivision) {
  EXPECT_EQ(Database::GetMatrixWidth(16, 8), 2u);
  EXPECT_EQ(Database::GetMatrixWidth(256, 32), 8u);
}

TEST(FrodoDatabaseTest, GetMatrixWidth_Static_RoundsUp) {
  EXPECT_EQ(Database::GetMatrixWidth(17, 8), 3u);
  EXPECT_EQ(Database::GetMatrixWidth(33, 8), 5u);
  EXPECT_EQ(Database::GetMatrixWidth(32, 32), 1u);
  EXPECT_EQ(Database::GetMatrixWidth(0, 32), 0u);
}

TEST(FrodoDatabaseTest, GetMatrixWidth_Static_PlaintextBitsZero) {
  EXPECT_EQ(Database::GetMatrixWidth(16, 0), 0u);
}

TEST(FrodoDatabaseTest, SwitchFmt_RoundtripIsIdentity) {
  auto db = MakeSmallDb();
  const auto before = db.EntriesForTest();
  db.SwitchFmt();
  const std::vector<std::vector<std::uint32_t>> row_form = {
      {1u, 2u}, {3u, 4u}, {5u, 6u},
  };
  EXPECT_EQ(db.EntriesForTest(), row_form);
  db.SwitchFmt();
  EXPECT_EQ(db.EntriesForTest(), before);
}

TEST(FrodoDatabaseTest, VecMult_HandComputed) {
  const auto db = MakeSmallDb();
  const std::vector<std::uint32_t> row = {2u, 4u, 6u};
  EXPECT_EQ(db.VecMult(row, 0), 44u);  // 1*2+3*4+5*6
  EXPECT_EQ(db.VecMult(row, 1), 56u);  // 2*2+4*4+6*6
}

TEST(FrodoDatabaseTest, GetRow_ReturnsExactClone) {
  const auto db = MakeSmallDb();
  EXPECT_EQ(db.GetRow(0),
            std::vector<std::uint32_t>({1u, 3u, 5u}));
  EXPECT_EQ(db.GetRow(1),
            std::vector<std::uint32_t>({2u, 4u, 6u}));
}

TEST(FrodoDatabaseTest, GetRow_OutOfRange_ReturnsEmpty) {
  const auto db = MakeSmallDb();
  EXPECT_TRUE(db.GetRow(2).empty());
  EXPECT_TRUE(db.GetRow(100).empty());
}

TEST(FrodoDatabaseTest, VecMult_AfterSwitchFmt_ColumnsMatchOriginalRows) {
  auto db = MakeSmallDb();
  db.SwitchFmt();
  const std::vector<std::uint32_t> row = {7u, 11u};
  EXPECT_EQ(db.VecMult(row, 0), 29u);  // {1,2}·{7,11} = 7+22
}

// ---- Chunk 3b tests (ConstructRows + Database::New) ------------

// Helper: encode a byte sequence as base64 via the same library
// Database::New uses, so fixtures stay self-consistent.
std::string B64(const std::vector<std::uint8_t>& bytes) {
  std::string s(bytes.begin(), bytes.end());
  return base64_encode(reinterpret_cast<const unsigned char*>(s.data()),
                       s.size());
}

TEST(FrodoConstructRowsTest, SingleByte_HandComputed) {
  // 1 element = {0xAB} (1 byte = 8 bits).
  // elem_size=8, plaintext_bits=8 -> row_width = 1.
  // bytes_to_bits_le(0xAB) = {1,1,0,1,0,1,0,1}.
  // bits_to_u32_le on all 8 = 0xAB = 171.
  std::vector<std::string> elements = {B64({0xABu})};
  std::vector<std::vector<std::uint32_t>> rows;
  std::string err;
  ASSERT_EQ(
      ConstructRows(elements, /*m=*/1, /*elem_size=*/8,
                    /*plaintext_bits=*/8, &rows, &err),
      retcode::SUCCESS)
      << err;
  ASSERT_EQ(rows.size(), 1u);
  ASSERT_EQ(rows[0].size(), 1u);
  EXPECT_EQ(rows[0][0], 0xABu);
}

TEST(FrodoConstructRowsTest, TwoBytes_FourBitPlaintextChunks) {
  // 1 element = {0xAB, 0xCD} (2 bytes = 16 bits).
  // elem_size=16, plaintext_bits=4 -> row_width = 4.
  // bytes_to_bits_le({0xAB,0xCD}):
  //   0xAB = 1010_1011 LSB-first -> 1,1,0,1,0,1,0,1
  //   0xCD = 1100_1101 LSB-first -> 1,0,1,1,0,0,1,1
  // Concat = 1,1,0,1, 0,1,0,1, 1,0,1,1, 0,0,1,1.
  // Chunks of 4:
  //   chunk 0 = {1,1,0,1} = 0xB (LSB-first: 1+2+0+8=11)
  //   chunk 1 = {0,1,0,1} = 0xA (LSB-first: 0+2+0+8=10)
  //   chunk 2 = {1,0,1,1} = 0xD (1+0+4+8=13)
  //   chunk 3 = {0,0,1,1} = 0xC (0+0+4+8=12)
  std::vector<std::string> elements = {B64({0xABu, 0xCDu})};
  std::vector<std::vector<std::uint32_t>> rows;
  std::string err;
  ASSERT_EQ(
      ConstructRows(elements, /*m=*/1, /*elem_size=*/16,
                    /*plaintext_bits=*/4, &rows, &err),
      retcode::SUCCESS)
      << err;
  ASSERT_EQ(rows.size(), 1u);
  EXPECT_EQ(rows[0],
            std::vector<std::uint32_t>({0xBu, 0xAu, 0xDu, 0xCu}));
}

TEST(FrodoConstructRowsTest, MultipleElements_RowOrder) {
  // 3 elements, each 1 byte. Hand-computed rows must come out in
  // the same order as elements.
  std::vector<std::string> elements = {
      B64({0x01u}),
      B64({0x80u}),  // 1000_0000 LSB-first = 0,0,0,0,0,0,0,1 = 0x80
      B64({0xFFu}),
  };
  std::vector<std::vector<std::uint32_t>> rows;
  std::string err;
  ASSERT_EQ(
      ConstructRows(elements, /*m=*/3, /*elem_size=*/8,
                    /*plaintext_bits=*/8, &rows, &err),
      retcode::SUCCESS)
      << err;
  ASSERT_EQ(rows.size(), 3u);
  EXPECT_EQ(rows[0][0], 0x01u);
  EXPECT_EQ(rows[1][0], 0x80u);
  EXPECT_EQ(rows[2][0], 0xFFu);
}

TEST(FrodoConstructRowsTest, SizeMismatch_Fails) {
  std::vector<std::string> elements = {B64({0x01u}), B64({0x02u})};
  std::vector<std::vector<std::uint32_t>> rows;
  std::string err;
  EXPECT_EQ(ConstructRows(elements, /*m=*/3, /*elem_size=*/8,
                          /*plaintext_bits=*/8, &rows, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("elements.size()=2"), std::string::npos) << err;
  EXPECT_NE(err.find("m=3"), std::string::npos) << err;
}

TEST(FrodoConstructRowsTest, NullOut_Fails) {
  std::vector<std::string> elements = {B64({0x01u})};
  std::string err;
  EXPECT_EQ(ConstructRows(elements, /*m=*/1, /*elem_size=*/8,
                          /*plaintext_bits=*/8, nullptr, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("out must be non-null"), std::string::npos)
      << err;
}

TEST(FrodoDatabaseNewTest, EndToEnd_TwoElementsColumnForm) {
  // 2 elements, each 1 byte: {0x12, 0x34}.
  // elem_size=8, plaintext_bits=8 -> row_width = 1.
  // construct_rows yields row-form [[0x12], [0x34]].
  // After swap_matrix_fmt, column-form = [[0x12, 0x34]].
  std::vector<std::string> elements = {B64({0x12u}), B64({0x34u})};
  Database db;  // default ctor placeholder
  std::string err;
  ASSERT_EQ(Database::New(elements, /*m=*/2, /*elem_size=*/8,
                          /*plaintext_bits=*/8, &db, &err),
            retcode::SUCCESS)
      << err;
  const std::vector<std::vector<std::uint32_t>> expected_col_form = {
      {0x12u, 0x34u},
  };
  EXPECT_EQ(db.EntriesForTest(), expected_col_form);
  EXPECT_EQ(db.GetMatrixHeight(), 2u);
  EXPECT_EQ(db.GetElemSize(), 8u);
  EXPECT_EQ(db.GetPlaintextBits(), 8u);
  EXPECT_EQ(db.GetMatrixWidthSelf(), 1u);
}

TEST(FrodoDatabaseNewTest, SwitchFmt_RecoversRowForm) {
  // Same fixture; after one SwitchFmt the row-form should match
  // [[0x12], [0x34]] (i.e. ConstructRows's output).
  std::vector<std::string> elements = {B64({0x12u}), B64({0x34u})};
  Database db;
  std::string err;
  ASSERT_EQ(Database::New(elements, 2, 8, 8, &db, &err), retcode::SUCCESS);
  db.SwitchFmt();
  const std::vector<std::vector<std::uint32_t>> row_form = {
      {0x12u}, {0x34u},
  };
  EXPECT_EQ(db.EntriesForTest(), row_form);
}

TEST(FrodoDatabaseNewTest, NullOutDb_Fails) {
  std::vector<std::string> elements = {B64({0x01u})};
  std::string err;
  EXPECT_EQ(Database::New(elements, 1, 8, 8, nullptr, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("out_db must be non-null"), std::string::npos)
      << err;
}

TEST(FrodoDatabaseNewTest, SizeMismatch_PropagatesConstructRowsError) {
  // m != elements.size() must surface ConstructRows's diagnostic
  // (the elements.size()=N and m=M tokens), proving the failure
  // path is wired.
  std::vector<std::string> elements = {B64({0x01u})};
  Database db;
  std::string err;
  EXPECT_EQ(Database::New(elements, 3, 8, 8, &db, &err),
            retcode::FAIL);
  EXPECT_NE(err.find("elements.size()=1"), std::string::npos) << err;
  EXPECT_NE(err.find("m=3"), std::string::npos) << err;
}



// ---- Chunk 3c tests (GetDbEntry) -------------------------------

// Upstream quirk: when elem_size % plaintext_bits == 0 the writer
// bytes_from_u32_slice treats the LAST entry as having `remainder
// = 0` bits, losing plaintext_bits worth of data. The reader
// construct_rows does NOT compensate (last chunk's end_bound==
// bits.len() takes the full remaining plaintext_bits). So the
// roundtrip is identity only when remainder != 0. Our port
// mirrors that byte-for-byte; the test fixtures below pick
// parameters that exercise the working configuration.
//
// elem_size=8, plaintext_bits=5: row_width=ceil(8/5)=2,
// remainder=8%5=3. row[0]=5 bits, row[1]=3 bits. 5+3==8 so the
// roundtrip is exact. This is the smallest single-byte fixture
// that avoids the exact-divide quirk.

TEST(FrodoGetDbEntryTest, Roundtrip_SingleByte_ParamsAvoidUpstreamQuirk) {
  const std::vector<std::uint8_t> orig_bytes = {0xABu};
  std::vector<std::string> elements = {B64(orig_bytes)};
  Database db;
  std::string err;
  ASSERT_EQ(Database::New(elements, /*m=*/1, /*elem_size=*/8,
                          /*plaintext_bits=*/5, &db, &err),
            retcode::SUCCESS)
      << err;
  const std::string got_b64 = db.GetDbEntry(0);
  const std::string got_bytes = base64_decode(got_b64);
  ASSERT_EQ(got_bytes.size(), orig_bytes.size());
  EXPECT_EQ(static_cast<std::uint8_t>(got_bytes[0]), orig_bytes[0]);
}

TEST(FrodoGetDbEntryTest, Roundtrip_TwoElements_OrderPreserved) {
  const std::vector<std::vector<std::uint8_t>> orig = {
      {0x12u}, {0x34u},
  };
  std::vector<std::string> elements = {B64(orig[0]), B64(orig[1])};
  Database db;
  std::string err;
  ASSERT_EQ(Database::New(elements, /*m=*/2, /*elem_size=*/8,
                          /*plaintext_bits=*/5, &db, &err),
            retcode::SUCCESS)
      << err;
  for (std::size_t i = 0; i < 2; ++i) {
    const std::string b64 = db.GetDbEntry(i);
    const std::string bytes = base64_decode(b64);
    ASSERT_EQ(bytes.size(), 1u) << "elem " << i;
    EXPECT_EQ(static_cast<std::uint8_t>(bytes[0]), orig[i][0])
        << "elem " << i;
  }
}

TEST(FrodoGetDbEntryTest, ExactDivide_MatchesUpstreamQuirk_LosesLastChunk) {
  // Anti-roundtrip test pinning upstream byte-for-byte fidelity:
  // elem_size=8, plaintext_bits=8 -> row_width=1, remainder=0.
  // bytes_from_u32_slice's last entry takes 0 bits, so the
  // entire byte is lost on encode. GetDbEntry must return the
  // empty string. If a well-meaning future fix changes this,
  // it breaks the upstream-fidelity invariant and this test
  // will catch it.
  const std::vector<std::uint8_t> orig_bytes = {0xABu};
  std::vector<std::string> elements = {B64(orig_bytes)};
  Database db;
  std::string err;
  ASSERT_EQ(Database::New(elements, /*m=*/1, /*elem_size=*/8,
                          /*plaintext_bits=*/8, &db, &err),
            retcode::SUCCESS)
      << err;
  EXPECT_EQ(db.GetDbEntry(0), std::string())
      << "Upstream quirk: exact-divide case drops the last chunk; "
      << "if this changes we have silently diverged from "
      << "brave-experiments/frodo-pir.";
}

TEST(FrodoGetDbEntryTest, OutOfRange_ReturnsEmptyString) {
  // Soft boundary (upstream panics on get_matrix_second_at).
  const auto db = MakeSmallDb();  // 2 columns of 3 entries each
  EXPECT_EQ(db.GetDbEntry(100), std::string());
}

}  // namespace
}  // namespace primihub::pir::frodo
