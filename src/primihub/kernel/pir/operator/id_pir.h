#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_ID_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_ID_PIR_H_

#include <string>
#include <vector>
#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

class IdPirOperator : public BasePirOperator {
 public:
  explicit IdPirOperator(const Options& options) : BasePirOperator(options) {}
  ~IdPirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;

 protected:
  retcode ClientSendRecv(const PirDataType& input, PirDataType* result);
  retcode ServerSendRecv(const PirDataType& input, PirDataType* result);
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_ID_PIR_H_
