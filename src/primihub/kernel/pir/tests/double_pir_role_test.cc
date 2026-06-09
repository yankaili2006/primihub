/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Role-aware OnExecute tests for DoublePirOperator — task 5.6 chunk 6
 * follow-up that wires BroadcastHint / ReceiveHint into the operator
 * via Options.hint_role ∈ {"", "primary", "secondary"}.
 *
 * Single-process behaviour stays covered by the existing
 * double_pir_test + double_pir_serialize_persist_test files; this one
 * exercises only the new wire-aware paths so they can fail
 * independently in CI.
 */
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_cache.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_link.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/util/network/link_context.h"

namespace primihub::pir {
namespace {

// In-process bus that records Sends and replays staged Recvs. Lets a
// primary operator's OnExecute Send hint bytes into the bus and a
// secondary's OnExecute Recv them out, both within the same gtest
// process.
class P2PBus : public network::LinkContext {
 public:
  std::shared_ptr<network::IChannel> getChannel(
      const primihub::Node&) override {
    return nullptr;
  }
  retcode Send(const std::string& key, const Node& dest,
               const std::string& data) override {
    sent_.push_back({key, dest.id(), data});
    queue_[Key(key, dest.id())] = data;
    return retcode::SUCCESS;
  }
  retcode Recv(const std::string& key, const Node& src,
               std::string* buf) override {
    auto it = queue_.find(Key(key, src.id()));
    if (it == queue_.end()) return retcode::FAIL;
    *buf = it->second;
    return retcode::SUCCESS;
  }

  struct Trace {
    std::string key;
    std::string dest_id;
    std::string data;
  };
  const std::vector<Trace>& sent() const { return sent_; }

 private:
  static std::string Key(const std::string& k, const std::string& id) {
    return k + "@" + id;
  }
  std::vector<Trace> sent_;
  std::unordered_map<std::string, std::string> queue_;
};

Node MakeNode(const std::string& id) {
  Node n;
  n.id_ = id;
  return n;
}

std::vector<std::string> MakeDb64() {
  std::vector<std::string> db;
  db.reserve(64);
  for (uint64_t i = 0; i < 64; ++i) {
    db.push_back(std::to_string((i * 13 + 7) & 0xFF));
  }
  return db;
}

class DoublePirRoleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    PirRegistry::EnsureRegistered();
    // Each test starts from a clean LRU + loaded_paths set so prior-
    // run state doesn't leak. Clear() also resets Hits/Misses.
    double_pir::HintCache::Instance().Clear();
  }
};

TEST_F(DoublePirRoleTest, PrimaryBroadcastsHintAfterCompute) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  P2PBus bus;
  Options opt;
  opt.self_party = "primary";
  opt.role = Role::SERVER;
  opt.hint_role = "primary";
  opt.link_ctx_ref = &bus;
  opt.peer_nodes = {MakeNode("secondary_a"), MakeNode("secondary_b")};

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);

  ASSERT_EQ(bus.sent().size(), 2u);
  EXPECT_EQ(bus.sent()[0].dest_id, "secondary_a");
  EXPECT_EQ(bus.sent()[1].dest_id, "secondary_b");
  EXPECT_EQ(bus.sent()[0].key, double_pir::kDefaultHintWireKey);
  EXPECT_EQ(bus.sent()[0].data, bus.sent()[1].data);
  // Sanity: payload is the real PHHB blob, not some FakeLinkContext
  // artifact — DeserializeHint should round-trip cleanly.
  double_pir::DoublePirHint roundtrip;
  EXPECT_EQ(double_pir::DeserializeHint(bus.sent()[0].data, &roundtrip,
                                         nullptr),
            retcode::SUCCESS);
  EXPECT_GT(roundtrip.info_after_setup.num, 0u);
}

TEST_F(DoublePirRoleTest, PrimaryWithoutPeersDoesNotBroadcast) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  P2PBus bus;
  Options opt;
  opt.self_party = "primary";
  opt.role = Role::SERVER;
  opt.hint_role = "primary";
  opt.link_ctx_ref = &bus;
  // peer_nodes empty — primary has no one to broadcast to.

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  EXPECT_TRUE(bus.sent().empty());
}

TEST_F(DoublePirRoleTest, PrimaryWithPeersRequiresLinkContext) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  Options opt;
  opt.self_party = "primary";
  opt.role = Role::SERVER;
  opt.hint_role = "primary";
  opt.link_ctx_ref = nullptr;  // operator must FAIL with peers set.
  opt.peer_nodes = {MakeNode("secondary_a")};

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST_F(DoublePirRoleTest, SecondaryReceivesHintInsteadOfComputing) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  // Stage a real hint into the bus by running a primary OnExecute
  // first — that's the cleanest way to populate the queue with a
  // hint shape that exactly matches the secondary's local (DB,
  // params) and validates the end-to-end wire path.
  P2PBus bus;
  Options primary_opt;
  primary_opt.self_party = "primary";
  primary_opt.role = Role::SERVER;
  primary_opt.hint_role = "primary";
  primary_opt.link_ctx_ref = &bus;
  primary_opt.peer_nodes = {MakeNode("secondary")};
  DoublePirOperator primary(primary_opt);
  PirDataType primary_in;
  primary_in["db_content"] = MakeDb64();
  primary_in["query_indices"] = {"0"};
  PirDataType primary_out;
  ASSERT_EQ(primary.OnExecute(primary_in, &primary_out), retcode::SUCCESS);
  ASSERT_EQ(bus.sent().size(), 1u);

  // Snapshot HintCache counters after the primary run — the secondary
  // must NOT bump Misses (HintGen::Compute would). Hits stays flat
  // too because secondary uses Put() rather than TryGet().
  const uint64_t hits_before  = double_pir::HintCache::Instance().Hits();
  const uint64_t misses_before = double_pir::HintCache::Instance().Misses();

  Options secondary_opt;
  secondary_opt.self_party = "secondary";
  secondary_opt.role = Role::SERVER;
  secondary_opt.hint_role = "secondary";
  secondary_opt.link_ctx_ref = &bus;
  secondary_opt.peer_nodes = {MakeNode("secondary")};  // self-Recv

  DoublePirOperator secondary(secondary_opt);
  PirDataType secondary_in;
  secondary_in["db_content"] = MakeDb64();
  secondary_in["query_indices"] = {"27"};
  PirDataType secondary_out;
  ASSERT_EQ(secondary.OnExecute(secondary_in, &secondary_out),
            retcode::SUCCESS);
  EXPECT_EQ(double_pir::HintCache::Instance().Hits(), hits_before);
  EXPECT_EQ(double_pir::HintCache::Instance().Misses(), misses_before);
  // Cache should have grown by one (the Put() at the bottom of
  // ReceiveHint handling).
  EXPECT_GE(double_pir::HintCache::Instance().Size(), 1u);
  ASSERT_EQ(secondary_out["recovered"].size(), 1u);
  // Correctness: byte at idx=27 in MakeDb64 is (27*13+7)&0xFF = 358 & 0xFF
  // = 102. Secondary should recover it exactly through the wire-shipped
  // hint, proving the protocol math + DB derivation is identical to
  // the primary's local view.
  EXPECT_EQ(secondary_out["recovered"][0], "102");
}

TEST_F(DoublePirRoleTest, SecondaryWithoutPeersFails) {
  P2PBus bus;
  Options opt;
  opt.self_party = "secondary";
  opt.role = Role::SERVER;
  opt.hint_role = "secondary";
  opt.link_ctx_ref = &bus;
  // peer_nodes empty — secondary has no primary to receive from.

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST_F(DoublePirRoleTest, SecondaryWithoutLinkContextFails) {
  Options opt;
  opt.self_party = "secondary";
  opt.role = Role::SERVER;
  opt.hint_role = "secondary";
  opt.link_ctx_ref = nullptr;
  opt.peer_nodes = {MakeNode("primary")};

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST_F(DoublePirRoleTest, SecondaryRecvFailureSurfacesAsFail) {
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  // Empty bus — Recv will return FAIL because nothing was staged.
  P2PBus bus;
  Options opt;
  opt.self_party = "secondary";
  opt.role = Role::SERVER;
  opt.hint_role = "secondary";
  opt.link_ctx_ref = &bus;
  opt.peer_nodes = {MakeNode("primary")};

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  EXPECT_EQ(op.OnExecute(input, &result), retcode::FAIL);
}

TEST_F(DoublePirRoleTest, EmptyHintRoleStillSingleProcess) {
  // The default-empty hint_role must keep the historical single-
  // process behaviour even when peer_nodes is non-empty (the latter
  // can be set by upstream config that hasn't migrated to hint_role
  // yet). No Sends, no Recvs.
  if (!core::kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs the kernel bridge";
  }
  P2PBus bus;
  Options opt;
  opt.self_party = "client";
  opt.role = Role::CLIENT;
  opt.hint_role = "";  // explicit default
  opt.link_ctx_ref = &bus;
  opt.peer_nodes = {MakeNode("p0")};  // present but unused

  DoublePirOperator op(opt);
  PirDataType input;
  input["db_content"] = MakeDb64();
  input["query_indices"] = {"27"};
  PirDataType result;
  ASSERT_EQ(op.OnExecute(input, &result), retcode::SUCCESS);
  EXPECT_TRUE(bus.sent().empty());
}

}  // namespace
}  // namespace primihub::pir
