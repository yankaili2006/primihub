/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/multi_peer_pir.h"

#include <glog/logging.h>

namespace primihub::pir {

retcode MultiPeerPirOperator::SendToAllPeers(const std::string& key,
                                             const std::string& data) {
  if (options_.peer_nodes.empty()) {
    LOG(ERROR) << "SendToAllPeers: peer_nodes empty";
    return retcode::FAIL;
  }
  auto* link = GetLinkContext();
  if (!link) {
    LOG(ERROR) << "SendToAllPeers: link context null";
    return retcode::FAIL;
  }
  for (size_t i = 0; i < options_.peer_nodes.size(); ++i) {
    if (has_stopped()) return retcode::FAIL;
    auto ret = link->Send(key, options_.peer_nodes[i], data);
    if (ret != retcode::SUCCESS) {
      LOG(ERROR) << "SendToAllPeers: peer " << i << " failed";
      return ret;
    }
  }
  return retcode::SUCCESS;
}

retcode MultiPeerPirOperator::RecvFromAllPeers(
    const std::string& key, std::vector<std::string>* replies) {
  if (!replies) return retcode::FAIL;
  replies->clear();
  if (options_.peer_nodes.empty()) {
    LOG(ERROR) << "RecvFromAllPeers: peer_nodes empty";
    return retcode::FAIL;
  }
  auto* link = GetLinkContext();
  if (!link) {
    LOG(ERROR) << "RecvFromAllPeers: link context null";
    return retcode::FAIL;
  }
  replies->resize(options_.peer_nodes.size());
  for (size_t i = 0; i < options_.peer_nodes.size(); ++i) {
    if (has_stopped()) return retcode::FAIL;
    auto ret = link->Recv(key, options_.peer_nodes[i], &(*replies)[i]);
    if (ret != retcode::SUCCESS) {
      LOG(ERROR) << "RecvFromAllPeers: peer " << i << " failed";
      return ret;
    }
  }
  return retcode::SUCCESS;
}

retcode MultiPeerPirOperator::SendRecvAllPeers(
    const std::string& key, const std::string& data,
    std::vector<std::string>* replies) {
  if (!replies) return retcode::FAIL;
  replies->clear();
  if (options_.peer_nodes.empty()) {
    LOG(ERROR) << "SendRecvAllPeers: peer_nodes empty";
    return retcode::FAIL;
  }
  auto* link = GetLinkContext();
  if (!link) {
    LOG(ERROR) << "SendRecvAllPeers: link context null";
    return retcode::FAIL;
  }
  replies->resize(options_.peer_nodes.size());
  for (size_t i = 0; i < options_.peer_nodes.size(); ++i) {
    if (has_stopped()) return retcode::FAIL;
    auto ret = link->SendRecv(key, options_.peer_nodes[i], data,
                              &(*replies)[i]);
    if (ret != retcode::SUCCESS) {
      LOG(ERROR) << "SendRecvAllPeers: peer " << i << " failed";
      return ret;
    }
  }
  return retcode::SUCCESS;
}

}  // namespace primihub::pir
