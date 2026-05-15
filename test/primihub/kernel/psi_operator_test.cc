// Copyright [2023] <PrimiHub>

#include "gtest/gtest.h"
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>

#include "src/primihub/kernel/psi/operator/common.h"
#include "src/primihub/kernel/psi/operator/base_psi.h"
#include "src/primihub/kernel/psi/operator/ecdh_psi.h"
#include "src/primihub/kernel/psi/operator/kkrt_psi.h"
#include "src/primihub/kernel/psi/operator/factory.h"
#include "src/primihub/common/common.h"

using namespace primihub::psi;

TEST(PsiCommonTest, PsiTypeEnum) {
  EXPECT_EQ(static_cast<int>(PsiType::ECDH), 0);
  EXPECT_EQ(static_cast<int>(PsiType::KKRT), 1);
  EXPECT_EQ(static_cast<int>(PsiType::TEE), 2);
}

TEST(PsiCommonTest, PsiResultTypeEnum) {
  EXPECT_EQ(static_cast<int>(PsiResultType::INTERSECTION), 0);
  EXPECT_EQ(static_cast<int>(PsiResultType::DIFFERENCE), 1);
}

// Test PsiResultType toggles difference calculation in BasePsiOperator::Execute
class TestPsiOperator : public BasePsiOperator {
 public:
  explicit TestPsiOperator(const Options& options)
      : BasePsiOperator(options) {}

  retcode OnExecute(const std::vector<std::string>& input,
                    std::vector<std::string>* result) override {
    // Simulate finding intersection: items starting with "common_"
    result->clear();
    for (const auto& item : input) {
      if (item.find("common_") == 0) {
        result->push_back(item);
      }
    }
    return retcode::SUCCESS;
  }
};

class PsiOperatorTest : public ::testing::Test {
 protected:
  struct SimpleLinkContext : network::LinkContext {
    SimpleLinkContext() : network::LinkContext() {}
  };

  void SetUp() override {
    party_info_["CLIENT"] = Node();
    party_info_["SERVER"] = Node();
  }

  Options MakeOptions(PsiResultType result_type) {
    Options opts;
    opts.self_party = "CLIENT";
    opts.party_info = party_info_;
    opts.psi_result_type = result_type;
    opts.proxy_node = Node();
    opts.link_ctx_ref = &link_ctx_;
    return opts;
  }

  std::map<std::string, Node> party_info_;
  SimpleLinkContext link_ctx_;
};

TEST_F(PsiOperatorTest, IntersectionResult) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "common_a", "unique_b", "common_c", "unique_d", "common_e"
  };
  std::vector<std::string> result;
  auto ret = psi_op.OnExecute(input, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "common_a");
  EXPECT_EQ(result[1], "common_c");
  EXPECT_EQ(result[2], "common_e");
}

TEST_F(PsiOperatorTest, DifferenceResult) {
  auto opts = MakeOptions(PsiResultType::DIFFERENCE);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "common_a", "unique_b", "common_c", "unique_d"
  };
  std::vector<std::string> result;
  auto ret = psi_op.Execute(input, false, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  // Items NOT starting with "common_" should be in difference
  EXPECT_EQ(result.size(), 2);
  EXPECT_EQ(result[0], "unique_b");
  EXPECT_EQ(result[1], "unique_d");
}

TEST_F(PsiOperatorTest, AllIntersectNoDifference) {
  auto opts = MakeOptions(PsiResultType::DIFFERENCE);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "common_a", "common_b", "common_c"
  };
  std::vector<std::string> result;
  auto ret = psi_op.Execute(input, false, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  // All items intersect, so difference is empty
  EXPECT_TRUE(result.empty());
}

TEST_F(PsiOperatorTest, NoIntersectAllDifference) {
  auto opts = MakeOptions(PsiResultType::DIFFERENCE);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "unique_a", "unique_b", "unique_c"
  };
  std::vector<std::string> result;
  auto ret = psi_op.Execute(input, false, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(result.size(), 3);
}

TEST_F(PsiOperatorTest, EmptyInput) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input;
  std::vector<std::string> result;
  auto ret = psi_op.OnExecute(input, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_TRUE(result.empty());
}

TEST_F(PsiOperatorTest, SingleElement) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {"common_only"};
  std::vector<std::string> result;
  auto ret = psi_op.OnExecute(input, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "common_only");
}

TEST_F(PsiOperatorTest, DuplicateInResult) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "common_a", "common_a", "common_a"
  };
  std::vector<std::string> result;
  auto ret = psi_op.OnExecute(input, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_EQ(result.size(), 3);
}

TEST_F(PsiOperatorTest, GetResultParallel) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  TestPsiOperator psi_op(opts);

  std::vector<std::string> input = {
    "data_0", "data_1", "data_2", "data_3", "data_4"
  };
  std::vector<uint64_t> indices = {0, 2, 4};
  std::vector<std::string> result;
  auto ret = psi_op.GetResult(input, indices, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "data_0");
  EXPECT_EQ(result[1], "data_2");
  EXPECT_EQ(result[2], "data_4");
}

TEST_F(PsiOperatorTest, GetResultEmptyIndices) {
  TestPsiOperator psi_op(MakeOptions(PsiResultType::INTERSECTION));

  std::vector<std::string> input = {"a", "b", "c"};
  std::vector<uint64_t> indices;
  std::vector<std::string> result;
  auto ret = psi_op.GetResult(input, indices, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_TRUE(result.empty());
}

TEST_F(PsiOperatorTest, FactoryCreateEcdh) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  auto psi_op = Factory::Create(PsiType::ECDH, opts);
  EXPECT_NE(psi_op, nullptr);
  EXPECT_NE(dynamic_cast<EcdhPsiOperator*>(psi_op.get()), nullptr);
}

TEST_F(PsiOperatorTest, FactoryCreateKkrt) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  auto psi_op = Factory::Create(PsiType::KKRT, opts);
  EXPECT_NE(psi_op, nullptr);
  EXPECT_NE(dynamic_cast<KkrtPsiOperator*>(psi_op.get()), nullptr);
}

TEST_F(PsiOperatorTest, EcdhOnExecuteEmptyInput) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  opts.self_party = "CLIENT";
  EcdhPsiOperator ecdh_op(opts);

  std::vector<std::string> empty_input;
  std::vector<std::string> result;
  auto ret = ecdh_op.OnExecute(empty_input, &result);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST_F(PsiOperatorTest, KkrtOnExecuteEmptyInput) {
  auto opts = MakeOptions(PsiResultType::INTERSECTION);
  opts.self_party = "CLIENT";
  KkrtPsiOperator kkrt_op(opts);

  std::vector<std::string> empty_input;
  std::vector<std::string> result;
  auto ret = kkrt_op.OnExecute(empty_input, &result);
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST_F(PsiOperatorTest, RoleValidation) {
  EXPECT_TRUE(RoleValidation::IsClient("CLIENT"));
  EXPECT_TRUE(RoleValidation::IsServer("SERVER"));
  EXPECT_FALSE(RoleValidation::IsClient("SERVER"));
  EXPECT_FALSE(RoleValidation::IsServer("CLIENT"));
  EXPECT_FALSE(RoleValidation::IsClient("UNKNOWN"));
  EXPECT_FALSE(RoleValidation::IsServer("UNKNOWN"));
}
