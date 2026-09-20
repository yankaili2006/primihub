/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/multi_peer_pir.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/util/network/link_context.h"

namespace primihub::pir {
namespace {

// Minimal in-process fake LinkContext for MultiPeer testing. Only the
// virtual methods MultiPeerPirOperator uses are overridden; the rest fall
// back to LinkContext's default no-op implementations (sufficient since
// the production code does not invoke them on this fake).
class FakeLinkContext : public network::LinkContext {
 public:
  std::shared_ptr<network::IChannel> getChannel(
      const primihub::Node&) override {
    return nullptr;
  }
  retcode Send(const std::string& key, const Node& dest,
               const std::string& data) override {
    sent_.push_back({key, dest.id(), data});
    return retcode::SUCCESS;
  }
  retcode Recv(const std::string& key, const Node& dest,
               std::string* buf) override {
    *buf = "recv_from_" + dest.id() + "_" + key;
    return retcode::SUCCESS;
  }
  retcode SendRecv(const std::string& key, const Node& dest,
                   const std::string& data, std::string* buf) override {
    sent_.push_back({key, dest.id(), data});
    *buf = "sr_reply_" + dest.id() + "_" + data;
    return retcode::SUCCESS;
  }

  struct Trace {
    std::string key;
    std::string dest_id;
    std::string data;
  };
  const std::vector<Trace>& sent() const { return sent_; }

 private:
  std::vector<Trace> sent_;
};

// Concrete subclass to access the protected helpers.
class TestableMultiPeerOp : public MultiPeerPirOperator {
 public:
  using MultiPeerPirOperator::MultiPeerPirOperator;
  using MultiPeerPirOperator::SendToAllPeers;
  using MultiPeerPirOperator::RecvFromAllPeers;
  using MultiPeerPirOperator::SendRecvAllPeers;
  using MultiPeerPirOperator::HasMinPeers;
  retcode OnExecute(const PirDataType&, PirDataType*) override {
    return retcode::SUCCESS;
  }
};

Node MakeNode(const std::string& id) {
  Node n;
  n.id_ = id;
  return n;
}

TEST(MultiPeerPirOperatorTest, HasMinPeersChecksSize) {
  Options opt;
  TestableMultiPeerOp op(opt);
  EXPECT_FALSE(op.HasMinPeers(1));
  opt.peer_nodes = {MakeNode("a")};
  TestableMultiPeerOp op1(opt);
  EXPECT_TRUE(op1.HasMinPeers(1));
  EXPECT_FALSE(op1.HasMinPeers(2));
  opt.peer_nodes = {MakeNode("a"), MakeNode("b")};
  TestableMultiPeerOp op2(opt);
  EXPECT_TRUE(op2.HasMinPeers(1));
  EXPECT_TRUE(op2.HasMinPeers(2));
  EXPECT_FALSE(op2.HasMinPeers(3));
}

TEST(MultiPeerPirOperatorTest, SendToAllPeersDeliversInOrder) {
  FakeLinkContext fake;
  Options opt;
  opt.link_ctx_ref = &fake;
  opt.peer_nodes = {MakeNode("peer0"), MakeNode("peer1")};
  TestableMultiPeerOp op(opt);

  auto ret = op.SendToAllPeers("my_key", "payload");
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(fake.sent().size(), 2u);
  EXPECT_EQ(fake.sent()[0].dest_id, "peer0");
  EXPECT_EQ(fake.sent()[1].dest_id, "peer1");
  EXPECT_EQ(fake.sent()[0].key, "my_key");
  EXPECT_EQ(fake.sent()[0].data, "payload");
}

TEST(MultiPeerPirOperatorTest, SendToAllPeersEmptyFails) {
  FakeLinkContext fake;
  Options opt;
  opt.link_ctx_ref = &fake;
  // peer_nodes empty
  TestableMultiPeerOp op(opt);
  auto ret = op.SendToAllPeers("k", "d");
  EXPECT_EQ(ret, retcode::FAIL);
}

TEST(MultiPeerPirOperatorTest, RecvFromAllPeersPopulatesReplies) {
  FakeLinkContext fake;
  Options opt;
  opt.link_ctx_ref = &fake;
  opt.peer_nodes = {MakeNode("alpha"), MakeNode("beta")};
  TestableMultiPeerOp op(opt);

  std::vector<std::string> replies;
  auto ret = op.RecvFromAllPeers("k", &replies);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(replies.size(), 2u);
  EXPECT_EQ(replies[0], "recv_from_alpha_k");
  EXPECT_EQ(replies[1], "recv_from_beta_k");
}

TEST(MultiPeerPirOperatorTest, SendRecvAllPeersRoundTrips) {
  FakeLinkContext fake;
  Options opt;
  opt.link_ctx_ref = &fake;
  opt.peer_nodes = {MakeNode("p0"), MakeNode("p1")};
  TestableMultiPeerOp op(opt);

  std::vector<std::string> replies;
  auto ret = op.SendRecvAllPeers("k", "ping", &replies);
  EXPECT_EQ(ret, retcode::SUCCESS);
  ASSERT_EQ(replies.size(), 2u);
  EXPECT_EQ(replies[0], "sr_reply_p0_ping");
  EXPECT_EQ(replies[1], "sr_reply_p1_ping");
  ASSERT_EQ(fake.sent().size(), 2u);
  EXPECT_EQ(fake.sent()[0].dest_id, "p0");
  EXPECT_EQ(fake.sent()[0].data, "ping");
}

TEST(MultiPeerPirOperatorTest, NullLinkContextFails) {
  Options opt;
  opt.link_ctx_ref = nullptr;
  opt.peer_nodes = {MakeNode("x")};
  TestableMultiPeerOp op(opt);
  std::vector<std::string> replies;
  EXPECT_EQ(op.RecvFromAllPeers("k", &replies), retcode::FAIL);
  EXPECT_EQ(op.SendToAllPeers("k", "d"), retcode::FAIL);
}

TEST(MultiPeerPirOperatorTest, StopBetweenPeersAborts) {
  FakeLinkContext fake;
  Options opt;
  opt.link_ctx_ref = &fake;
  opt.peer_nodes = {MakeNode("a"), MakeNode("b")};
  TestableMultiPeerOp op(opt);
  op.set_stop();
  std::vector<std::string> replies;
  EXPECT_EQ(op.RecvFromAllPeers("k", &replies), retcode::FAIL);
}

}  // namespace
}  // namespace primihub::pir
