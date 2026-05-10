#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/primihub/util/network/mock_channel.h"
#include "src/primihub/kernel/psi/operator/base_psi.h"
#include "src/primihub/kernel/psi/operator/factory.h"

using namespace primihub;
using namespace primihub::network;
using namespace primihub::psi;
using ::testing::Return;
using ::testing::_;

class TestablePsiOperator : public BasePsiOperator {
 public:
  TestablePsiOperator(const Options& options) : BasePsiOperator(options) {}
  retcode OnExecute(const std::vector<std::string>& input,
                    std::vector<std::string>* result) override {
    *result = input;
    return retcode::SUCCESS;
  }
  using BasePsiOperator::GetResult;
  using BasePsiOperator::Execute;
  using BasePsiOperator::BroadcastPsiResult;
  using BasePsiOperator::BroadcastPartyList;
};

class BasePsiTest : public ::testing::Test {
 protected:
  MockLinkContext mock_ctx_;
  std::shared_ptr<MockChannel> mock_channel_;
  Node node_;

  void SetUp() override {
    mock_channel_ = std::make_shared<MockChannel>();
    node_.id_ = "party0";
    node_.ip_ = "127.0.0.1";
    node_.port_ = 50050;
    ON_CALL(mock_ctx_, getChannel(::testing::An<const primihub::Node&>()))
        .WillByDefault(Return(mock_channel_));
  }
};

TEST_F(BasePsiTest, GetResult_Basic) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  TestablePsiOperator op(opts);
  std::vector<std::string> input = {"a", "b", "c", "d", "e"};
  std::vector<uint64_t> indices = {0, 2, 4};
  std::vector<std::string> result;
  auto ret = op.GetResult(input, indices, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(result.size(), 3);
  EXPECT_EQ(result[0], "a");
  EXPECT_EQ(result[1], "c");
  EXPECT_EQ(result[2], "e");
}

TEST_F(BasePsiTest, GetResult_EmptyIndices) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;

  TestablePsiOperator op(opts);
  std::vector<std::string> input = {"a", "b", "c"};
  std::vector<uint64_t> indices;
  std::vector<std::string> result;
  op.GetResult(input, indices, &result);
  EXPECT_TRUE(result.empty());
}

TEST_F(BasePsiTest, GetResult_SingleElement) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;

  TestablePsiOperator op(opts);
  std::vector<std::string> input = {"only"};
  std::vector<uint64_t> indices = {0};
  std::vector<std::string> result;
  op.GetResult(input, indices, &result);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0], "only");
}

TEST_F(BasePsiTest, Execute_Difference_PartialOverlap) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;
  opts.psi_result_type = PsiResultType::DIFFERENCE;

  class DiffPsiOp : public TestablePsiOperator {
   public:
    DiffPsiOp(const Options& opts) : TestablePsiOperator(opts) {}
    retcode OnExecute(const std::vector<std::string>& input,
                      std::vector<std::string>* result) override {
      *result = {"a"};
      return retcode::SUCCESS;
    }
  };

  DiffPsiOp op(opts);
  std::vector<std::string> input = {"a", "b", "c"};
  std::vector<std::string> result;
  auto ret = op.Execute(input, false, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(result.size(), 2);
}

TEST_F(BasePsiTest, Execute_Difference_AllOverlap) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;
  opts.psi_result_type = PsiResultType::DIFFERENCE;

  TestablePsiOperator op(opts);
  std::vector<std::string> input = {"a", "b"};
  std::vector<std::string> result = {"a", "b"};
  auto ret = op.Execute(input, false, &result);
  EXPECT_EQ(ret, retcode::SUCCESS);
  EXPECT_TRUE(result.empty());
}

TEST_F(BasePsiTest, BroadcastCallsChannelSend) {
  Node node1;
  node1.id_ = "party1";
  node1.ip_ = "127.0.0.1";
  node1.port_ = 50051;

  auto mock_channel2 = std::make_shared<MockChannel>();
  EXPECT_CALL(*mock_channel2, send_view("default", ::testing::_))
      .Times(1)
      .WillOnce(Return(primihub::retcode::SUCCESS));

  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = PARTY_CLIENT;
  opts.party_info[PARTY_SERVER] = node1;

  EXPECT_CALL(mock_ctx_, getChannel(::testing::An<const primihub::Node&>()))
      .WillRepeatedly(Return(mock_channel2));

  TestablePsiOperator op(opts);
  std::vector<std::string> result = {"result1", "result2"};
  auto ret = op.BroadcastPsiResult(&result);
  EXPECT_EQ(ret, retcode::SUCCESS);
}

TEST_F(BasePsiTest, Factory_CreatesPsiTypes) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  auto ecdh = Factory::Create(PsiType::ECDH, opts);
  ASSERT_NE(ecdh, nullptr);

  auto kkrt = Factory::Create(PsiType::KKRT, opts);
  ASSERT_NE(kkrt, nullptr);
}

TEST_F(BasePsiTest, Factory_UnknownTypeReturnsNull) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = "party0";
  opts.party_info["party0"] = node_;

  auto op = Factory::Create(static_cast<PsiType>(99), opts);
  EXPECT_EQ(op, nullptr);
}
