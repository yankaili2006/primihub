/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_REGISTRY_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_REGISTRY_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"

namespace primihub::pir {

class BasePirOperator;
struct Options;

class PirRegistry {
 public:
  using Creator = std::function<std::unique_ptr<BasePirOperator>(const Options&)>;

  static PirRegistry& Instance();

  // Bootstrap entry that the main binary MUST call before serving any task,
  // AFTER InitGoogleLogging. Drains the pending-registration list populated by
  // PirRegistrar instances at static-initialization time and installs them
  // into the real registry. Until EnsureRegistered runs, ListAlgorithms /
  // Create / GetCapabilities will all behave as if no algorithms are present.
  //
  // Splitting registration into a "pending" phase (TU-scope statics, glog
  // not yet ready) and a "drain" phase (called from main after glog init)
  // avoids the static-initialization-order fiasco that previously crashed
  // any binary linking id_pir_operator before initializing logging.
  static void EnsureRegistered();

  bool Register(const std::string& algo, Creator creator,
                const PirCapabilities& caps);

  std::unique_ptr<BasePirOperator> Create(const std::string& algo,
                                          const Options& options);

  // Returns nullptr when the algorithm is not registered.
  const PirCapabilities* GetCapabilities(const std::string& algo) const;

  std::vector<std::string> ListAlgorithms() const;

  // Internal — called by PirRegistrar at static-initialization time. Never
  // touches glog. Stores the registration in a TU-scope pending list that
  // EnsureRegistered() drains later.
  static void AddPending(const std::string& algo, Creator creator,
                         const PirCapabilities& caps);

 private:
  PirRegistry() = default;
  PirRegistry(const PirRegistry&) = delete;
  PirRegistry& operator=(const PirRegistry&) = delete;

  mutable std::mutex mu_;
  std::map<std::string, std::pair<Creator, PirCapabilities>> registry_;
};

// Template helper used at namespace scope inside each algorithm's .cc file:
//   namespace {
//     PirRegistrar<SpiralPirOperator> reg_("spiral", caps);
//   }
// Pushes the registration onto a TU-scope pending list at static-init time;
// the entry is moved into the real registry by PirRegistry::EnsureRegistered.
template <typename T>
class PirRegistrar {
 public:
  PirRegistrar(const std::string& algo, const PirCapabilities& caps) {
    PirRegistry::AddPending(
        algo,
        [](const Options& opt) -> std::unique_ptr<BasePirOperator> {
          return std::unique_ptr<BasePirOperator>(new T(opt));
        },
        caps);
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_REGISTRY_H_
