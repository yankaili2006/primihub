#include "gtest/gtest.h"
#include <arrow/api.h>
#include "src/primihub/kernel/psi/util.h"

using namespace primihub::psi;
using namespace primihub;

class PsiUtilTest : public ::testing::Test {
 protected:
  PsiCommonUtil util_;
};

TEST_F(PsiUtilTest, IsValidDataType_Numeric32) {
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::INT32));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::INT16));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::UINT32));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::UINT8));
}

TEST_F(PsiUtilTest, IsValidDataType_Numeric64) {
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::INT64));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::UINT64));
}

TEST_F(PsiUtilTest, IsValidDataType_String) {
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::STRING));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::BINARY));
  EXPECT_TRUE(util_.IsValidDataType(arrow::Type::FIXED_SIZE_BINARY));
}

TEST_F(PsiUtilTest, IsValidDataType_Invalid) {
  EXPECT_FALSE(util_.IsValidDataType(arrow::Type::BOOL));
  EXPECT_FALSE(util_.IsValidDataType(arrow::Type::DATE32));
  EXPECT_FALSE(util_.IsValidDataType(arrow::Type::TIMESTAMP));
  EXPECT_FALSE(util_.IsValidDataType(arrow::Type::NA));
}

TEST_F(PsiUtilTest, IsNumeric) {
  EXPECT_TRUE(util_.isNumeric(arrow::Type::INT32));
  EXPECT_TRUE(util_.isNumeric(arrow::Type::INT64));
  EXPECT_FALSE(util_.isNumeric(arrow::Type::STRING));
}

TEST_F(PsiUtilTest, IsNumeric32) {
  EXPECT_TRUE(util_.isNumeric32Type(arrow::Type::INT32));
  EXPECT_TRUE(util_.isNumeric32Type(arrow::Type::UINT16));
  EXPECT_FALSE(util_.isNumeric32Type(arrow::Type::INT64));
  EXPECT_FALSE(util_.isNumeric32Type(arrow::Type::STRING));
}

TEST_F(PsiUtilTest, IsNumeric64) {
  EXPECT_TRUE(util_.isNumeric64Type(arrow::Type::INT64));
  EXPECT_TRUE(util_.isNumeric64Type(arrow::Type::UINT64));
  EXPECT_FALSE(util_.isNumeric64Type(arrow::Type::INT32));
}

TEST_F(PsiUtilTest, IsString) {
  EXPECT_TRUE(util_.isString(arrow::Type::STRING));
  EXPECT_TRUE(util_.isString(arrow::Type::BINARY));
  EXPECT_FALSE(util_.isString(arrow::Type::INT32));
}

TEST_F(PsiUtilTest, ValidationDataColum_Match) {
  EXPECT_TRUE(util_.validationDataColum({0, 1, 2}, 3));
}

TEST_F(PsiUtilTest, ValidationDataColum_NoMatch) {
  EXPECT_FALSE(util_.validationDataColum({0, 1}, 3));
}

TEST_F(PsiUtilTest, ValidationDataColum_Empty) {
  EXPECT_TRUE(util_.validationDataColum({}, 0));
}
