#include "gtest/gtest.h"
#include <string>
#include "src/primihub/kernel/psi/operator/factory.h"
#include "src/primihub/kernel/psi/operator/common.h"

using namespace primihub::psi;

TEST(psi_protocol, factory_create_ecdh) {
  Options opts;
  opts.party_info.clear();
  opts.self_party = "CLIENT";
  auto op = Factory::Create(PsiType::ECDH, opts);
  EXPECT_NE(op, nullptr);
}

TEST(psi_protocol, factory_create_kkrt) {
  Options opts;
  opts.party_info.clear();
  opts.self_party = "CLIENT";
  auto op = Factory::Create(PsiType::KKRT, opts);
  EXPECT_NE(op, nullptr);
}

TEST(psi_protocol, enum_values) {
  EXPECT_EQ(static_cast<int>(PsiType::ECDH), 0);
  EXPECT_EQ(static_cast<int>(PsiType::KKRT), 1);
  EXPECT_EQ(static_cast<int>(PsiType::TEE), 2);
  EXPECT_EQ(static_cast<int>(PsiResultType::INTERSECTION), 0);
  EXPECT_EQ(static_cast<int>(PsiResultType::DIFFERENCE), 1);
}
