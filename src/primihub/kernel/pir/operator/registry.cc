/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/registry.h"

#include <glog/logging.h>
#include <utility>
#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

PirRegistry& PirRegistry::Instance() {
  static PirRegistry singleton;
  return singleton;
}

void PirRegistry::EnsureRegistered() {
  // Force initialization of this TU and any TU referenced from the binary's
  // dep graph. Algorithms register via static initializers in their own TUs;
  // by depending on this function from main(), the linker keeps those TUs.
  (void)Instance();
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
  LOG(INFO) << "PirRegistry: registered '" << algo << "'";
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
