/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Unit tests for ColMajorMatrix (chunk g-1 of the task 7.1
 * flat-buffer refactor). The tests pin the column-major stride
 * convention + the basic semantics that consumer chunks g-2..g-5
 * rely on. They are intentionally tight: chunk g-1 is a pure
 * additive change — anything more elaborate belongs to the
 * consumer-migration chunks.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_flat_matrix.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace primihub::pir::frodo {
namespace {

TEST(FrodoFlatMatrixTest, EmptyMatrix_IsEmpty) {
  ColMajorMatrix m;
  EXPECT_EQ(m.height(), 0u);
  EXPECT_EQ(m.width(), 0u);
  EXPECT_TRUE(m.empty());
  EXPECT_EQ(m.total_u32s(), 0u);
}

TEST(FrodoFlatMatrixTest, SizedCtor_AllocatesAndZeroes) {
  // 3 rows x 4 columns => 12 u32s, all zero.
  ColMajorMatrix m(/*height=*/3, /*width=*/4);
  EXPECT_EQ(m.height(), 3u);
  EXPECT_EQ(m.width(), 4u);
  ASSERT_EQ(m.total_u32s(), 12u);
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      EXPECT_EQ(m.at(c, r), 0u)
          << "default ctor should zero-init element (c=" << c
          << ", r=" << r << ")";
    }
  }
}

TEST(FrodoFlatMatrixTest, RoundtripAt) {
  // Round-trip every element through at() then re-read. Catches
  // any silent stride miscompute.
  ColMajorMatrix m(/*height=*/5, /*width=*/7);
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      m.at(c, r) = static_cast<std::uint32_t>(c * 100u + r);
    }
  }
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      EXPECT_EQ(m.at(c, r),
                static_cast<std::uint32_t>(c * 100u + r));
    }
  }
}

TEST(FrodoFlatMatrixTest, ColumnData_Sequential) {
  // Pins the column-major stride: column_data(c)[r] must match
  // at(c, r). If this test passes for a non-trivial shape every
  // future consumer (chunks g-2..g-5) can rely on column_data(c)
  // returning a contiguous height-length u32 slice — that is
  // exactly the shape VecMultU32U32 wants.
  ColMajorMatrix m(/*height=*/6, /*width=*/4);
  for (std::size_t c = 0; c < m.width(); ++c) {
    for (std::size_t r = 0; r < m.height(); ++r) {
      m.at(c, r) = static_cast<std::uint32_t>(c * 31u + r * 7u + 1u);
    }
  }
  for (std::size_t c = 0; c < m.width(); ++c) {
    const std::uint32_t* cd = m.column_data(c);
    ASSERT_NE(cd, nullptr) << "column_data must be non-null at c=" << c;
    for (std::size_t r = 0; r < m.height(); ++r) {
      EXPECT_EQ(cd[r], m.at(c, r))
          << "column-major stride broken at (c=" << c
          << ", r=" << r << ")";
    }
  }
}

TEST(FrodoFlatMatrixTest, EqualityWorks) {
  // Same shape + same content -> equal.
  ColMajorMatrix a(/*height=*/3, /*width=*/2);
  ColMajorMatrix b(/*height=*/3, /*width=*/2);
  for (std::size_t c = 0; c < a.width(); ++c) {
    for (std::size_t r = 0; r < a.height(); ++r) {
      const std::uint32_t v =
          static_cast<std::uint32_t>(c * 2u + r * 3u);
      a.at(c, r) = v;
      b.at(c, r) = v;
    }
  }
  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);

  // Same shape, different content.
  b.at(0, 0) = b.at(0, 0) + 1u;
  EXPECT_FALSE(a == b);
  EXPECT_TRUE(a != b);

  // Different shape.
  ColMajorMatrix c2(/*height=*/2, /*width=*/3);
  EXPECT_FALSE(a == c2);
}

TEST(FrodoFlatMatrixTest, NoInitCtor_HasSameShape) {
  // The NoInit ctor must have the same shape + total_u32s as the
  // regular ctor — the only difference is the documented contract
  // that the caller will overwrite before reading. Today the
  // backing storage is also zero-initialised by libstdc++'s
  // vector::resize for uint32_t, but the test does NOT assert
  // zeros so a future allocator switch that genuinely skips the
  // zero-fill does not break this test.
  ColMajorMatrix m(/*height=*/5, /*width=*/6, ColMajorMatrix::NoInit{});
  EXPECT_EQ(m.height(), 5u);
  EXPECT_EQ(m.width(), 6u);
  EXPECT_EQ(m.total_u32s(), 30u);
  EXPECT_FALSE(m.empty());
}

}  // namespace
}  // namespace primihub::pir::frodo
