#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/primihub/util/network/mock_channel.h"
#include "src/primihub/kernel/psi/operator/ecdh_psi.h"
#include "src/primihub/kernel/psi/operator/factory.h"

using namespace primihub;
using namespace primihub::network;
using namespace primihub::psi;
using ::testing::Return;
using ::testing::_;

class EcdhPsiMockTest : public ::testing::Test {
 protected:
  MockLinkContext mock_ctx_;
  std::shared_ptr<MockChannel> mock_channel_;
  Node client_node_;
  Node server_node_;

  void SetUp() override {
    mock_channel_ = std::make_shared<MockChannel>();
    client_node_.id_ = PARTY_CLIENT;
    client_node_.ip_ = "127.0.0.1";
    client_node_.port_ = 50050;
    server_node_.id_ = PARTY_SERVER;
    server_node_.ip_ = "127.0.0.1";
    server_node_.port_ = 50051;

    ON_CALL(mock_ctx_, getChannel(_))
        .WillByDefault(Return(mock_channel_));
    ON_CALL(mock_ctx_, Send_str(_, _, _))
        .WillByDefault(Return(retcode::SUCCESS));
    ON_CALL(mock_ctx_, Recv_from(_, _, _))
        .WillByDefault(Return(retcode::SUCCESS));
    ON_CALL(mock_ctx_, Recv_str(_, _))
        .WillByDefault(Return(retcode::SUCCESS));
  }
};

TEST_F(EcdhPsiMockTest, FactoryCreatesEcdhPsi) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = PARTY_CLIENT;
  opts.party_info[PARTY_SERVER] = server_node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  auto op = Factory::Create(PsiType::ECDH, opts);
  ASSERT_NE(op, nullptr);
}

TEST_F(EcdhPsiMockTest, ClientSendsInitParam) {
  EXPECT_CALL(mock_ctx_, Send_str("default", _, _))
      .Times(::testing::AtLeast(1));

  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = PARTY_CLIENT;
  opts.party_info[PARTY_SERVER] = server_node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  EcdhPsiOperator op(opts);
  std::vector<std::string> input = {"item1", "item2", "item3"};
  std::vector<std::string> result;
  op.OnExecute(input, &result);
}

TEST_F(EcdhPsiMockTest, ServerReceivesInitParam) {
  EXPECT_CALL(mock_ctx_, Recv_from("default", _, _))
      .Times(::testing::AtLeast(1));

  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = PARTY_SERVER;
  opts.party_info[PARTY_CLIENT] = client_node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  EcdhPsiOperator op(opts);
  std::vector<std::string> input = {"item1", "item2"};
  std::vector<std::string> result;
  op.OnExecute(input, &result);
}

TEST_F(EcdhPsiMockTest, KkrtFactory) {
  Options opts;
  opts.link_ctx_ref = &mock_ctx_;
  opts.self_party = PARTY_CLIENT;
  opts.party_info[PARTY_SERVER] = server_node_;
  opts.psi_result_type = PsiResultType::INTERSECTION;

  auto op = Factory::Create(PsiType::KKRT, opts);
  ASSERT_NE(op, nullptr);
}
