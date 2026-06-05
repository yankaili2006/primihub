/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/registry.h"

#include <glog/logging.h>
#include <utility>
#include <vector>
#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

namespace {

// Pending registrations populated at static-init time by PirRegistrar.
// Held by raw pointer so that the storage is constant-initialized (the
// pointer itself is zero-initialized before any dynamic init runs); the
// std::vector body is lazily allocated on first AddPending call, which is
// safe because libc malloc is set up by ld.so before any C++ static init.
//
// IMPORTANT: do not touch glog from inside AddPending — at static-init time
// the logging subsystem may not yet be ready, and earlier versions of this
// file crashed binaries that linked algorithms before main() ran
// InitGoogleLogging().
struct PendingEntry {
  std::string algo;
  PirRegistry::Creator creator;
  PirCapabilities caps;
};

std::vector<PendingEntry>*& PendingList() {
  // Function-local static. Construction is deferred to first call, but the
  // local 'pending' pointer is statically allocated, so multiple calls
  // (even reentrant from static init) always return the same address.
  static std::vector<PendingEntry>* pending = new std::vector<PendingEntry>();
  return pending;
}

}  // namespace

PirRegistry& PirRegistry::Instance() {
  static PirRegistry singleton;
  return singleton;
}

void PirRegistry::AddPending(const std::string& algo, Creator creator,
                             const PirCapabilities& caps) {
  // Avoid glog — runs at static-init time.
  PendingList()->push_back({algo, std::move(creator), caps});
}

void PirRegistry::EnsureRegistered() {
  // Safe to log from here: callers MUST have already done
  // google::InitGoogleLogging(argv[0]).
  auto* pending = PendingList();
  if (!pending || pending->empty()) {
    return;
  }
  auto& reg = Instance();
  std::vector<PendingEntry> drain;
  drain.swap(*pending);
  for (auto& e : drain) {
    reg.Register(e.algo, std::move(e.creator), e.caps);
  }
}

bool PirRegistry::Register(const std::string& algo, Creator creator,
                           const PirCapabilities& caps) {
  if (algo.empty()) {
    LOG(ERROR) << "PirRegistry::Register: empty algorithm name";
    return false;
  }
  auto check = caps.Check();
  if (!check.empty()) {
    LOG(ERROR) << "PirRegistry::Register: capabilities for '" << algo
               << "' inconsistent: " << check;
    return false;
  }
  std::lock_guard<std::mutex> lock(mu_);
  auto [it, inserted] = registry_.emplace(
      algo, std::make_pair(std::move(creator), caps));
  if (!inserted) {
    LOG(WARNING) << "PirRegistry::Register: algorithm '" << algo
                 << "' already registered, keeping first";
    return false;
  }
  VLOG(2) << "PirRegistry: registered '" << algo << "'";
  return true;
}

std::unique_ptr<BasePirOperator> PirRegistry::Create(const std::string& algo,
                                                     const Options& options) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registry_.find(algo);
  if (it == registry_.end()) {
    LOG(ERROR) << "PIR algorithm not registered: " << algo;
    return nullptr;
  }
  return it->second.first(options);
}

const PirCapabilities* PirRegistry::GetCapabilities(
    const std::string& algo) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registry_.find(algo);
  if (it == registry_.end()) return nullptr;
  return &it->second.second;
}

std::vector<std::string> PirRegistry::ListAlgorithms() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::vector<std::string> names;
  names.reserve(registry_.size());
  for (const auto& kv : registry_) {
    names.push_back(kv.first);
  }
  return names;
}

}  // namespace primihub::pir
