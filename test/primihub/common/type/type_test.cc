#include "gtest/gtest.h"
#include "src/primihub/common/type.h"

using namespace primihub;

TEST(ColumnDtypeTest, ToString) {
  EXPECT_EQ(columnDtypeToString(ColumnDtype::STRING), "STRING");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::INTEGER), "INT64");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::DOUBLE), "FP64");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::BOOLEAN), "BOOLEAN");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::LONG), "INT64");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::ENUM), "ENUM");
  EXPECT_EQ(columnDtypeToString(ColumnDtype::UNKNOWN), "UNKNOWN TYPE");
}

TEST(ColumnDtypeTest, FromInteger) {
  ColumnDtype type;
  columnDtypeFromInteger(0, type);
  EXPECT_EQ(type, ColumnDtype::STRING);
  columnDtypeFromInteger(1, type);
  EXPECT_EQ(type, ColumnDtype::INTEGER);
  columnDtypeFromInteger(2, type);
  EXPECT_EQ(type, ColumnDtype::DOUBLE);
  columnDtypeFromInteger(3, type);
  EXPECT_EQ(type, ColumnDtype::LONG);
  columnDtypeFromInteger(4, type);
  EXPECT_EQ(type, ColumnDtype::ENUM);
  columnDtypeFromInteger(5, type);
  EXPECT_EQ(type, ColumnDtype::BOOLEAN);
}

TEST(ColumnDtypeTest, FromInteger_Default) {
  ColumnDtype type;
  columnDtypeFromInteger(99, type);
  EXPECT_EQ(type, ColumnDtype::UNKNOWN);
}
