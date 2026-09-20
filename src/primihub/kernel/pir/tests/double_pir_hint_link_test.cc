/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_link.h"

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/util/network/link_context.h"

namespace primihub::pir::double_pir {
namespace {

// In-process fake LinkContext that records Send traffic + lets tests
// stage canned Recv replies keyed by (wire_key, peer_id). The
// production code only invokes Send / Recv; default LinkContext
// implementations for the rest stay no-op.
class FakeLinkContext : public network::LinkContext {
 public:
  std::shared_ptr<network::IChannel> getChannel(
      const primihub::Node&) override {
    return nullptr;
  }
  retcode Send(const std::string& key, const Node& dest,
               const std::string& data) override {
    sent_.push_back({key, dest.id(), data});
    return send_should_succeed_ ? retcode::SUCCESS : retcode::FAIL;
  }
  retcode Recv(const std::string& key, const Node& src,
               std::string* buf) override {
    if (!recv_should_succeed_) return retcode::FAIL;
    auto it = recv_map_.find(MakeKey(key, src.id()));
    if (it == recv_map_.end()) return retcode::FAIL;
    *buf = it->second;
    return retcode::SUCCESS;
  }

  struct Trace {
    std::string key;
    std::string dest_id;
    std::string data;
  };
  const std::vector<Trace>& sent() const { return sent_; }
  void SetSendSucceeds(bool v) { send_should_succeed_ = v; }
  void SetRecvSucceeds(bool v) { recv_should_succeed_ = v; }
  void StageRecv(const std::string& key, const std::string& peer_id,
                 const std::string& payload) {
    recv_map_[MakeKey(key, peer_id)] = payload;
  }

 private:
  static std::string MakeKey(const std::string& k, const std::string& id) {
    return k + "@" + id;
  }
  std::vector<Trace> sent_;
  std::unordered_map<std::string, std::string> recv_map_;
  bool send_should_succeed_ = true;
  bool recv_should_succeed_ = true;
};

Node MakeNode(const std::string& id) {
  Node n;
  n.id_ = id;
  return n;
}

// Build a syntactically-valid minimal hint: five 1x1 matrices and a
// default DBinfo. SerializeHint only needs wire-level shape, so a 1x1
// matrix per slot is enough to round-trip the BroadcastHint → Recv →
// DeserializeHint path. Cells get sentinel values so tests can detect
// silent matrix swaps.
DoublePirHint MakeMinimalHint() {
  DoublePirHint hint;
  hint.A1 = core::Matrix(1, 1);
  hint.A2 = core::Matrix(1, 1);
  hint.H1_squished = core::Matrix(1, 1);
  hint.A2_copy_transposed = core::Matrix(1, 1);
  hint.H2_msg = core::Matrix(1, 1);
  hint.A1.Set(0, 0, 7);
  hint.A2.Set(0, 0, 13);
  hint.H1_squished.Set(0, 0, 21);
  hint.A2_copy_transposed.Set(0, 0, 29);
  hint.H2_msg.Set(0, 0, 31);
  hint.info_after_setup.num = 64;
  hint.info_after_setup.row_length = 8;
  hint.info_after_setup.packing = 1;
  hint.info_after_setup.ne = 1;
  hint.info_after_setup.x = 1;
  hint.info_after_setup.p = 929;
  hint.info_after_setup.logq = 32;
  hint.info_after_setup.basis = 10;
  hint.info_after_setup.squishing = 3;
  hint.info_after_setup.cols = 8;
  return hint;
}

TEST(DoublePirHintLinkTest, BroadcastHintNoPeersIsNoop) {
  FakeLinkContext fake;
  auto hint = MakeMinimalHint();
  std::string err;
  auto rc = BroadcastHint(&fake, /*peers=*/{}, hint, kDefaultHintWireKey,
                          &err);
  EXPECT_EQ(rc, retcode::SUCCESS);
  EXPECT_TRUE(fake.sent().empty());
  EXPECT_TRUE(err.empty());
}

TEST(DoublePirHintLinkTest, BroadcastHintNoPeersAcceptsNullLink) {
  // Caller can wire BroadcastHint unconditionally even when there is
  // no link context yet — the empty-peers branch short-circuits before
  // touching the link.
  auto hint = MakeMinimalHint();
  std::string err;
  auto rc = BroadcastHint(/*link=*/nullptr, /*peers=*/{}, hint,
                          kDefaultHintWireKey, &err);
  EXPECT_EQ(rc, retcode::SUCCESS);
  EXPECT_TRUE(err.empty());
}

TEST(DoublePirHintLinkTest, BroadcastHintNullLinkWithPeersFails) {
  auto hint = MakeMinimalHint();
  std::string err;
  std::vector<Node> peers = {MakeNode("p0")};
  auto rc = BroadcastHint(/*link=*/nullptr, peers, hint,
                          kDefaultHintWireKey, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_FALSE(err.empty());
}

TEST(DoublePirHintLinkTest, BroadcastHintShipsIdenticalBytesToAllPeers) {
  FakeLinkContext fake;
  auto hint = MakeMinimalHint();
  std::vector<Node> peers = {MakeNode("p0"), MakeNode("p1"), MakeNode("p2")};
  std::string err;
  auto rc = BroadcastHint(&fake, peers, hint, "double_pir.hint.v1", &err);
  EXPECT_EQ(rc, retcode::SUCCESS) << err;
  ASSERT_EQ(fake.sent().size(), 3u);
  EXPECT_EQ(fake.sent()[0].dest_id, "p0");
  EXPECT_EQ(fake.sent()[1].dest_id, "p1");
  EXPECT_EQ(fake.sent()[2].dest_id, "p2");
  EXPECT_EQ(fake.sent()[0].key, "double_pir.hint.v1");
  EXPECT_EQ(fake.sent()[0].data, fake.sent()[1].data);
  EXPECT_EQ(fake.sent()[1].data, fake.sent()[2].data);
  // Round-trip: deserialize the payload back into a hint and check
  // sentinel cells to prove the bytes ARE the real serialization
  // (not some FakeLinkContext artifact).
  DoublePirHint roundtrip;
  EXPECT_EQ(DeserializeHint(fake.sent()[0].data, &roundtrip, nullptr),
            retcode::SUCCESS);
  EXPECT_EQ(roundtrip.A1.Get(0, 0), 7u);
  EXPECT_EQ(roundtrip.H2_msg.Get(0, 0), 31u);
}

TEST(DoublePirHintLinkTest, BroadcastHintStopsOnFirstSendFailure) {
  FakeLinkContext fake;
  fake.SetSendSucceeds(false);
  auto hint = MakeMinimalHint();
  std::vector<Node> peers = {MakeNode("p0"), MakeNode("p1")};
  std::string err;
  auto rc = BroadcastHint(&fake, peers, hint, kDefaultHintWireKey, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_FALSE(err.empty());
  // Only one Send attempted before the helper short-circuited.
  ASSERT_EQ(fake.sent().size(), 1u);
}

TEST(DoublePirHintLinkTest, BroadcastHintUsesDefaultKeyWhenUnspecified) {
  FakeLinkContext fake;
  auto hint = MakeMinimalHint();
  std::vector<Node> peers = {MakeNode("p0")};
  std::string err;
  auto rc = BroadcastHint(&fake, peers, hint);  // key default-arg
  EXPECT_EQ(rc, retcode::SUCCESS) << err;
  ASSERT_EQ(fake.sent().size(), 1u);
  EXPECT_EQ(fake.sent()[0].key, kDefaultHintWireKey);
}

TEST(DoublePirHintLinkTest, ReceiveHintNullArgsFail) {
  FakeLinkContext fake;
  DoublePirHint hint;
  std::string err;
  EXPECT_EQ(ReceiveHint(/*link=*/nullptr, MakeNode("p0"), &hint,
                        kDefaultHintWireKey, &err),
            retcode::FAIL);
  EXPECT_FALSE(err.empty());
  err.clear();
  EXPECT_EQ(ReceiveHint(&fake, MakeNode("p0"), /*hint_out=*/nullptr,
                        kDefaultHintWireKey, &err),
            retcode::FAIL);
  EXPECT_FALSE(err.empty());
}

TEST(DoublePirHintLinkTest, ReceiveHintRoundTripsThroughFake) {
  FakeLinkContext fake;
  auto hint = MakeMinimalHint();
  std::string blob;
  ASSERT_EQ(SerializeHint(hint, &blob, nullptr), retcode::SUCCESS);
  fake.StageRecv(kDefaultHintWireKey, "p_primary", blob);

  DoublePirHint received;
  std::string err;
  auto rc = ReceiveHint(&fake, MakeNode("p_primary"), &received,
                        kDefaultHintWireKey, &err);
  EXPECT_EQ(rc, retcode::SUCCESS) << err;
  EXPECT_EQ(received.A1.Get(0, 0), 7u);
  EXPECT_EQ(received.A2.Get(0, 0), 13u);
  EXPECT_EQ(received.H1_squished.Get(0, 0), 21u);
  EXPECT_EQ(received.A2_copy_transposed.Get(0, 0), 29u);
  EXPECT_EQ(received.H2_msg.Get(0, 0), 31u);
  EXPECT_EQ(received.info_after_setup.num, 64u);
  EXPECT_EQ(received.info_after_setup.p, 929u);
}

TEST(DoublePirHintLinkTest, ReceiveHintRecvFailureSurfaces) {
  FakeLinkContext fake;
  fake.SetRecvSucceeds(false);
  DoublePirHint hint;
  std::string err;
  auto rc = ReceiveHint(&fake, MakeNode("p0"), &hint, kDefaultHintWireKey,
                        &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_FALSE(err.empty());
}

TEST(DoublePirHintLinkTest, ReceiveHintCorruptBytesFails) {
  FakeLinkContext fake;
  fake.StageRecv(kDefaultHintWireKey, "p0", "not a valid PHHB blob");
  DoublePirHint hint;
  std::string err;
  auto rc = ReceiveHint(&fake, MakeNode("p0"), &hint, kDefaultHintWireKey,
                        &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_FALSE(err.empty());
}

TEST(DoublePirHintLinkTest, RoundTripPrimaryToSecondaryThroughBus) {
  // End-to-end: primary calls BroadcastHint, secondary calls
  // ReceiveHint through the same in-process bus. We model the bus as
  // a queue keyed by (wire_key, dest_id). For Recv, src.id() is the
  // queue key — so primary Sends to "secondary" and secondary Recvs
  // FROM "secondary" (its own queue), matching how the production
  // LinkContext routes by dest tag.
  class P2PBus : public network::LinkContext {
   public:
    std::shared_ptr<network::IChannel> getChannel(
        const primihub::Node&) override { return nullptr; }
    retcode Send(const std::string& key, const Node& dest,
                 const std::string& data) override {
      q_[MakeKey(key, dest.id())] = data;
      return retcode::SUCCESS;
    }
    retcode Recv(const std::string& key, const Node& src,
                 std::string* buf) override {
      auto it = q_.find(MakeKey(key, src.id()));
      if (it == q_.end()) return retcode::FAIL;
      *buf = it->second;
      return retcode::SUCCESS;
    }
   private:
    static std::string MakeKey(const std::string& k, const std::string& id) {
      return k + "@" + id;
    }
    std::unordered_map<std::string, std::string> q_;
  };
  P2PBus bus;

  auto primary_hint = MakeMinimalHint();
  std::vector<Node> peers_from_primary = {MakeNode("secondary")};
  std::string err;
  ASSERT_EQ(BroadcastHint(&bus, peers_from_primary, primary_hint,
                          kDefaultHintWireKey, &err),
            retcode::SUCCESS) << err;

  DoublePirHint secondary_hint;
  ASSERT_EQ(ReceiveHint(&bus, MakeNode("secondary"), &secondary_hint,
                        kDefaultHintWireKey, &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(secondary_hint.A1.Get(0, 0), primary_hint.A1.Get(0, 0));
  EXPECT_EQ(secondary_hint.A2.Get(0, 0), primary_hint.A2.Get(0, 0));
  EXPECT_EQ(secondary_hint.H1_squished.Get(0, 0),
            primary_hint.H1_squished.Get(0, 0));
  EXPECT_EQ(secondary_hint.A2_copy_transposed.Get(0, 0),
            primary_hint.A2_copy_transposed.Get(0, 0));
  EXPECT_EQ(secondary_hint.H2_msg.Get(0, 0), primary_hint.H2_msg.Get(0, 0));
  EXPECT_EQ(secondary_hint.info_after_setup.num,
            primary_hint.info_after_setup.num);
  EXPECT_EQ(secondary_hint.info_after_setup.p,
            primary_hint.info_after_setup.p);
  EXPECT_EQ(secondary_hint.info_after_setup.squishing,
            primary_hint.info_after_setup.squishing);
}

}  // namespace
}  // namespace primihub::pir::double_pir
