#include "gtest/gtest.h"
#include <arrow/api.h>
#include "src/primihub/util/arrow_wrapper_util.h"

using namespace primihub::arrow_wrapper::util;

TEST(ArrowWrapperTest, SqlType2ArrowType_Integer) {
  int arrow_type;
  auto ret = SqlType2ArrowType("INTEGER", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::INT64);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Int) {
  int arrow_type;
  auto ret = SqlType2ArrowType("INT", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::INT32);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Text) {
  int arrow_type;
  auto ret = SqlType2ArrowType("TEXT", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::STRING);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Varchar) {
  int arrow_type;
  auto ret = SqlType2ArrowType("VARCHAR(255)", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::STRING);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Double) {
  int arrow_type;
  auto ret = SqlType2ArrowType("DOUBLE", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::DOUBLE);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Float) {
  int arrow_type;
  auto ret = SqlType2ArrowType("FLOAT", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::FLOAT);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_Unknown_DefaultsToString) {
  int arrow_type;
  auto ret = SqlType2ArrowType("UNKNOWN_TYPE", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::STRING);
}

TEST(ArrowWrapperTest, SqlType2ArrowType_CaseInsensitive) {
  int arrow_type;
  auto ret = SqlType2ArrowType("integer", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::INT64);

  ret = SqlType2ArrowType("InTeGeR", &arrow_type);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
  EXPECT_EQ(arrow_type, arrow::Type::INT64);
}

TEST(ArrowWrapperTest, MakeArrowDataType_Int32) {
  auto type = MakeArrowDataType(arrow::Type::INT32);
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->Equals(arrow::int32()));
}

TEST(ArrowWrapperTest, MakeArrowDataType_String) {
  auto type = MakeArrowDataType(arrow::Type::STRING);
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->Equals(arrow::utf8()));
}

TEST(ArrowWrapperTest, MakeArrowDataType_Bool) {
  auto type = MakeArrowDataType(arrow::Type::BOOL);
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->Equals(arrow::boolean()));
}

TEST(ArrowWrapperTest, MakeArrowDataType_DefaultToString) {
  auto type = MakeArrowDataType(999);
  ASSERT_NE(type, nullptr);
  EXPECT_TRUE(type->Equals(arrow::utf8()));
}

TEST(ArrowWrapperTest, Int32ArrowArrayBuilder_Basic) {
  auto arr = Int32ArrowArrayBuilder({"1", "2", "3"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 3);
}

TEST(ArrowWrapperTest, Int64ArrowArrayBuilder_Basic) {
  auto arr = Int64ArrowArrayBuilder({"100", "200"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 2);
}

TEST(ArrowWrapperTest, FloatArrowArrayBuilder_Basic) {
  auto arr = FloatArrowArrayBuilder({"1.5", "2.5"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 2);
}

TEST(ArrowWrapperTest, DoubleArrowArrayBuilder_Basic) {
  auto arr = DoubleArrowArrayBuilder({"3.14", "2.71"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 2);
}

TEST(ArrowWrapperTest, StringArrowArrayBuilder_Basic) {
  auto arr = StringArrowArrayBuilder({"hello", "world"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 2);
}

TEST(ArrowWrapperTest, MakeArrowArray_ByType) {
  auto arr = MakeArrowArray(arrow::Type::INT32, {"10", "20", "30"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 3);
}

TEST(ArrowWrapperTest, MakeArrowArray_DefaultToString) {
  auto arr = MakeArrowArray(999, {"fallback"});
  ASSERT_NE(arr, nullptr);
  ASSERT_EQ(arr->length(), 1);
}
