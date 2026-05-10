#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test/primihub/util/network/mock_channel.h"

using namespace primihub::network;
using ::testing::Return;
using ::testing::_;

TEST(MockChannelTest, MockSubmitTask) {
  auto channel = std::make_shared<MockChannel>();
  primihub::rpc::PushTaskRequest req;
  primihub::rpc::PushTaskReply reply;
  EXPECT_CALL(*channel, submitTask(_, _))
      .WillOnce(Return(primihub::retcode::SUCCESS));
  auto ret = channel->submitTask(req, &reply);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
}

TEST(MockChannelTest, MockKillTask) {
  auto channel = std::make_shared<MockChannel>();
  primihub::rpc::KillTaskRequest req;
  primihub::rpc::KillTaskResponse reply;
  EXPECT_CALL(*channel, killTask(_, _))
      .WillOnce(Return(primihub::retcode::SUCCESS));
  auto ret = channel->killTask(req, &reply);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
}

TEST(MockChannelTest, MockFetchTaskStatus) {
  auto channel = std::make_shared<MockChannel>();
  primihub::rpc::TaskContext ctx;
  primihub::rpc::TaskStatusReply reply;
  EXPECT_CALL(*channel, fetchTaskStatus(_, _))
      .WillOnce(Return(primihub::retcode::SUCCESS));
  auto ret = channel->fetchTaskStatus(ctx, &reply);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
}

TEST(MockChannelTest, MockNewDataset) {
  auto channel = std::make_shared<MockChannel>();
  primihub::rpc::NewDatasetRequest req;
  primihub::rpc::NewDatasetResponse reply;
  EXPECT_CALL(*channel, NewDataset(_, _))
      .WillOnce(Return(primihub::retcode::SUCCESS));
  auto ret = channel->NewDataset(req, &reply);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
}

TEST(MockChannelTest, MockCheckSendCompleteStatus) {
  auto channel = std::make_shared<MockChannel>();
  EXPECT_CALL(*channel, CheckSendCompleteStatus("key", 5))
      .WillOnce(Return(primihub::retcode::SUCCESS));
  auto ret = channel->CheckSendCompleteStatus("key", 5);
  EXPECT_EQ(ret, primihub::retcode::SUCCESS);
}

TEST(MockChannelTest, MockLinkContext) {
  MockLinkContext mock_ctx;
  auto mock_channel = std::make_shared<MockChannel>();
  primihub::Node node;
  node.id_ = "test_node";

  EXPECT_CALL(mock_ctx, getChannel(_))
      .WillRepeatedly(Return(mock_channel));

  auto retrieved = mock_ctx.getChannel(node);
  ASSERT_NE(retrieved, nullptr);
}
