/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_MULTI_PEER_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_MULTI_PEER_PIR_H_

#include <cstdint>
#include <string>
#include <vector>
#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// Base class for PIR algorithms with min_servers >= 2 (e.g. DoublePIR,
// TreePIR). Provides sequential broadcast/gather helpers over
// options_.peer_nodes. The 2-peer case is the common one; the helpers
// scale to N peers but iterate sequentially — adequate for the bandwidth-
// and CPU-bound regime where each per-peer call is the dominant cost.
class MultiPeerPirOperator : public BasePirOperator {
 public:
  using BasePirOperator::BasePirOperator;

 protected:
  // Sends the same payload to every peer under `key`. SUCCESS only when
  // every peer's Send returned SUCCESS.
  retcode SendToAllPeers(const std::string& key, const std::string& data);

  // Receives one payload from each peer (in options_.peer_nodes order)
  // under `key`. replies->size() == peer_nodes.size() on SUCCESS.
  retcode RecvFromAllPeers(const std::string& key,
                           std::vector<std::string>* replies);

  // Sends `data` to every peer and gathers each peer's response under
  // the same key. replies->size() == peer_nodes.size() on SUCCESS.
  retcode SendRecvAllPeers(const std::string& key, const std::string& data,
                           std::vector<std::string>* replies);

  // Returns true iff peer_nodes.size() >= min_peers. Algorithms call this
  // in OnExecute before doing work to short-circuit when misconfigured.
  bool HasMinPeers(uint32_t min_peers) const {
    return options_.peer_nodes.size() >= min_peers;
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_MULTI_PEER_PIR_H_
