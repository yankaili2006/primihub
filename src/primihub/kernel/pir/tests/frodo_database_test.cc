/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_database_test — verifies the Database struct port (the
 * deterministic, base64-free subset of upstream src/db.rs's
 * `pub struct Database`). Covers the data ctor, in-place format
 * switch, dot product against a column, row accessor, and the
 * matrix-width helper (static + self).
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace primihub::pir::frodo {
namespace {

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
  // 16 bits at 8 bits/entry -> 2 entries; 256 bits at 32 -> 8.
  EXPECT_EQ(Database::GetMatrixWidth(16, 8), 2u);
  EXPECT_EQ(Database::GetMatrixWidth(256, 32), 8u);
}

TEST(FrodoDatabaseTest, GetMatrixWidth_Static_RoundsUp) {
  // 17 bits at 8 bits/entry -> ceil(17/8) = 3.
  // 33 bits at 8 -> 5; 32 at 32 -> 1; 0 at 32 -> 0.
  EXPECT_EQ(Database::GetMatrixWidth(17, 8), 3u);
  EXPECT_EQ(Database::GetMatrixWidth(33, 8), 5u);
  EXPECT_EQ(Database::GetMatrixWidth(32, 32), 1u);
  EXPECT_EQ(Database::GetMatrixWidth(0, 32), 0u);
}

TEST(FrodoDatabaseTest, GetMatrixWidth_Static_PlaintextBitsZero) {
  // Soft boundary — upstream would divide by zero.
  EXPECT_EQ(Database::GetMatrixWidth(16, 0), 0u);
}

TEST(FrodoDatabaseTest, SwitchFmt_RoundtripIsIdentity) {
  auto db = MakeSmallDb();
  const auto before = db.EntriesForTest();
  db.SwitchFmt();
  // After one switch we should see the row-form: {{1,2},{3,4},{5,6}}.
  const std::vector<std::vector<std::uint32_t>> row_form = {
      {1u, 2u},
      {3u, 4u},
      {5u, 6u},
  };
  EXPECT_EQ(db.EntriesForTest(), row_form);
  db.SwitchFmt();
  // Back to column-form.
  EXPECT_EQ(db.EntriesForTest(), before);
}

TEST(FrodoDatabaseTest, VecMult_HandComputed) {
  const auto db = MakeSmallDb();
  // entries[col 0] = {1, 3, 5}; row = {2, 4, 6}.
  // dot = 1*2 + 3*4 + 5*6 = 2 + 12 + 30 = 44.
  const std::vector<std::uint32_t> row = {2u, 4u, 6u};
  EXPECT_EQ(db.VecMult(row, 0), 44u);
  // entries[col 1] = {2, 4, 6}; same row dot = 2*2 + 4*4 + 6*6
  //                                          = 4 + 16 + 36 = 56.
  EXPECT_EQ(db.VecMult(row, 1), 56u);
}

TEST(FrodoDatabaseTest, GetRow_ReturnsExactClone) {
  const auto db = MakeSmallDb();
  // entries is in column form, so GetRow(0) returns column 0:
  //   {1, 3, 5}.
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
  // Pre-switch entries: column-form {{1,3,5}, {2,4,6}}.
  // Post-switch entries: row-form {{1,2}, {3,4}, {5,6}}.
  // After switch, entries[0] = {1,2}; dot with {7, 11}
  //   = 1*7 + 2*11 = 7 + 22 = 29.
  auto db = MakeSmallDb();
  db.SwitchFmt();
  const std::vector<std::uint32_t> row = {7u, 11u};
  EXPECT_EQ(db.VecMult(row, 0), 29u);
}

}  // namespace
}  // namespace primihub::pir::frodo
