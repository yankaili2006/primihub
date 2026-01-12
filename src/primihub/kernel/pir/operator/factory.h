// "Copyright [2023] <PrimiHub>"
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
#include <glog/logging.h>
#include <memory>
#include "src/primihub/kernel/pir/common.h"
#ifdef MICROSOFT_APSI
#include "src/primihub/kernel/pir/operator/keyword_pir.h"
#endif
namespace primihub::pir {
class Factory {
 public:
  static std::unique_ptr<BasePirOperator> Create(PirType pir_type,
      const Options& options) {
    std::unique_ptr<BasePirOperator> operator_ptr{nullptr};
    switch (pir_type) {
    case PirType::ID_PIR:
      LOG(ERROR) << "Unimplement";
      break;
    case PirType::KEY_PIR:
#ifdef MICROSOFT_APSI
      operator_ptr = std::make_unique<KeywordPirOperator>(options);
#else
      LOG(ERROR) << "KEY_PIR not supported: Microsoft APSI is disabled";
#endif
      break;
    default:
      LOG(ERROR) << "unknown pir operator: " << static_cast<int>(pir_type);
      break;
    }
    return operator_ptr;
  }
};
}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FACTORY_H_
