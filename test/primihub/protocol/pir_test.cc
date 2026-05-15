#include "gtest/gtest.h"
#include <string>
#include "src/primihub/kernel/pir/operator/factory.h"
#include "src/primihub/kernel/pir/common.h"

using namespace primihub::pir;

TEST(pir_protocol, enum_values) {
  EXPECT_EQ(static_cast<int>(PirType::ID_PIR), 0);
  EXPECT_EQ(static_cast<int>(PirType::KEY_PIR), 1);
}
