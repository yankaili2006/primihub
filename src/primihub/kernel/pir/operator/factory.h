// "Copyright [2023] <PrimiHub>"
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
#include <glog/logging.h>
#include <memory>
#include <string>
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

// Backward-compatible factory. New callers SHOULD use
// PirRegistry::Instance().Create("<algo>", options) directly; this shim
// translates legacy PirType enums into algorithm name lookups.
class Factory {
 public:
  static std::unique_ptr<BasePirOperator> Create(PirType pir_type,
      const Options& options) {
    const std::string algo = LegacyNameFor(pir_type);
    if (algo.empty()) {
      LOG(ERROR) << "unknown legacy PirType: " << static_cast<int>(pir_type);
      return nullptr;
    }
    auto op = PirRegistry::Instance().Create(algo, options);
    if (!op) {
      LOG(ERROR) << "legacy PirType " << ToString(pir_type)
                 << " (algo '" << algo << "') not available; "
                 << "check that the algorithm was linked into the binary";
    }
    return op;
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
