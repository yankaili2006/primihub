/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_link.h"

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"

namespace primihub::pir::double_pir {

retcode BroadcastHint(network::LinkContext* link,
                      const std::vector<Node>& peers,
                      const DoublePirHint& hint,
                      const std::string& key,
                      std::string* err) {
  if (peers.empty()) {
    // Single-process mode — caller can call this unconditionally
    // without gating on peer_nodes.size().
    return retcode::SUCCESS;
  }
  if (link == nullptr) {
    const std::string msg = "BroadcastHint: link context is null";
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return retcode::FAIL;
  }
  std::string blob;
  std::string serr;
  auto rc = SerializeHint(hint, &blob, &serr);
  if (rc != retcode::SUCCESS) {
    const std::string msg = "BroadcastHint: SerializeHint failed: " + serr;
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return retcode::FAIL;
  }
  for (size_t i = 0; i < peers.size(); ++i) {
    auto sret = link->Send(key, peers[i], blob);
    if (sret != retcode::SUCCESS) {
      const std::string msg = "BroadcastHint: Send to peer index " +
                              std::to_string(i) + " (id=" + peers[i].id() +
                              ") failed";
      if (err) *err = msg;
      LOG(ERROR) << msg;
      return sret;
    }
  }
  VLOG(3) << "BroadcastHint: shipped " << blob.size() << " bytes to "
          << peers.size() << " peer(s) under key=" << key;
  return retcode::SUCCESS;
}

retcode ReceiveHint(network::LinkContext* link, const Node& peer,
                    DoublePirHint* hint_out, const std::string& key,
                    std::string* err) {
  if (hint_out == nullptr) {
    const std::string msg = "ReceiveHint: hint_out is null";
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return retcode::FAIL;
  }
  if (link == nullptr) {
    const std::string msg = "ReceiveHint: link context is null";
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return retcode::FAIL;
  }
  std::string blob;
  auto rret = link->Recv(key, peer, &blob);
  if (rret != retcode::SUCCESS) {
    const std::string msg = "ReceiveHint: Recv from peer id=" + peer.id() +
                            " under key=" + key + " failed";
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return rret;
  }
  std::string derr;
  auto dret = DeserializeHint(blob, hint_out, &derr);
  if (dret != retcode::SUCCESS) {
    const std::string msg = "ReceiveHint: DeserializeHint failed: " + derr;
    if (err) *err = msg;
    LOG(ERROR) << msg;
    return retcode::FAIL;
  }
  VLOG(3) << "ReceiveHint: deserialized " << blob.size() << " bytes from peer id="
          << peer.id();
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::double_pir
