/*
 * Copyright (c) 2023 by PrimiHub
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BASE_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BASE_PIR_H_
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/util/network/link_context.h"
namespace primihub::pir {
using LinkContext = network::LinkContext;
using IndexType = int32_t;

// Options is the per-operator config. The historical single-peer field
// (peer_node) is preserved via getter/setter for callers that still pass
// one peer; new algorithms (e.g. DoublePIR) use peer_nodes directly.
struct Options {
  LinkContext* link_ctx_ref{nullptr};
  std::map<std::string, Node> party_info;
  std::string self_party;
  std::string code;
  Role role;
  // online
  bool use_cache{false};
  // offline task
  bool generate_db{false};
  std::string db_path;
  // Multi-peer support (introduced 2026-06 by pir-multi-peer-link capability).
  // peer_nodes[0] is the conventional first peer (legacy single-peer position).
  std::vector<Node> peer_nodes;
  Node proxy_node;

  // P0 additions for the multi-algorithm framework.
  Backend preferred_backend{Backend::AUTO};
  bool assume_non_colluding{false};
  std::string hint_path;

  // Backward-compatible setter: legacy code wrote `options.peer_node = X`.
  // Existing call sites should be migrated to peer_nodes; this helper keeps
  // single-peer callers working without source changes when accessed via
  // helpers below.
  void set_peer_node(const Node& n) {
    if (peer_nodes.empty()) {
      peer_nodes.push_back(n);
    } else {
      peer_nodes[0] = n;
    }
  }
  const Node& peer_node() const {
    static const Node empty{};
    if (peer_nodes.empty()) return empty;
    return peer_nodes[0];
  }
};

class BasePirOperator {
 public:
  explicit BasePirOperator(const Options& options) : options_(options) {}
  virtual ~BasePirOperator() = default;
  retcode Execute(const PirDataType& input, PirDataType* result);
  virtual retcode OnExecute(const PirDataType& input, PirDataType* result) = 0;
  void set_stop() {stop_.store(true);}

 protected:
  bool has_stopped() {
    return stop_.load(std::memory_order::memory_order_relaxed);
  }
  std::string PartyName() {return options_.self_party;}
  Role role() {return options_.role;}
  LinkContext* GetLinkContext() {return options_.link_ctx_ref;}
  // PeerNode preserved for back-compat. Returns peer_nodes[0] or a stale
  // default-constructed Node when no peer is set; callers wanting multi-peer
  // semantics should iterate options_.peer_nodes directly.
  Node PeerNode() {
    if (!options_.peer_nodes.empty()) return options_.peer_nodes[0];
    return Node{};
  }
  const std::vector<Node>& PeerNodes() const {return options_.peer_nodes;}
  Node& ProxyNode() {return options_.proxy_node;}
  std::string PackageCountKey(const std::string& /*request_id*/) {
    return "pack_count";
  }

 protected:
  std::atomic<bool> stop_{false};
  Options options_;
  std::string key_{"pir_key"};
  std::string response_key_{"response_pir_key"};
  std::string key_task_end_{"pir_task_end"};
  std::string loop_num_key_{"loop_num_key"};
};
}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BASE_PIR_H_
