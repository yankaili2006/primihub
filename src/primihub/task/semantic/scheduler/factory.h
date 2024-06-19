// Copyright [2023] <PRimihub>
#ifndef SRC_PRIMIHUB_TASK_SEMANTIC_SCHEDULER_FACTORY_H_
#define SRC_PRIMIHUB_TASK_SEMANTIC_SCHEDULER_FACTORY_H_
#include <memory>
#include "src/primihub/task/semantic/scheduler/scheduler.h"

#ifdef PY_TASK_ENABLED
#include "src/primihub/task/semantic/scheduler/fl_scheduler.h"
#endif  // PY_TASK_ENABLED
#ifdef MPC_TASK_ENABLED
#include "src/primihub/task/semantic/scheduler/mpc_scheduler.h"
#include "src/primihub/task/semantic/scheduler/aby3_scheduler.h"
#endif  // MPC_TASK_ENABLED

#include "src/primihub/task/semantic/scheduler/tee_scheduler.h"
#ifdef PIR_TASK_ENABLED
#include "src/primihub/task/semantic/scheduler/pir_scheduler.h"
#endif // PIR_TASK_ENABLED
#include "src/primihub/protos/common.pb.h"
#include <glog/logging.h>
#include "src/primihub/common/value_check_util.h"
namespace primihub::task {
class SchedulerFactory {
 public:
  static std::unique_ptr<VMScheduler> CreateScheduler(const rpc::Task& task_config) {
    std::unique_ptr<VMScheduler> scheduler{nullptr};
    auto language = task_config.language();
    switch (language) {
#ifdef PY_TASK_ENABLED
    case rpc::Language::PYTHON:
      scheduler = SchedulerFactory::CreatePythonScheduler(task_config);
      break;
#endif  // PY_TASK_ENABLED
    case rpc::Language::PROTO:
      scheduler = SchedulerFactory::CreateProtoScheduler(task_config);
      break;
    default: {
      std::stringstream ss;
      ss << "Unsupported language type: " << rpc::Language_Name(language);
      RaiseException(ss.str());
    }
    }
    return scheduler;
  }
#ifdef PY_TASK_ENABLED
  static std::unique_ptr<VMScheduler> CreatePythonScheduler(const rpc::Task& task_config) {
    return std::make_unique<FLScheduler>();
  }
#endif // PY_TASK_ENABLED
  static std::unique_ptr<VMScheduler> CreateProtoScheduler(const rpc::Task& task_config) {
    std::unique_ptr<VMScheduler> scheduler{nullptr};
    auto task_type = task_config.type();
    switch (task_type) {
#ifdef MPC_TASK_ENABLED
    case rpc::TaskType::ACTOR_TASK:
      scheduler = SchedulerFactory::CreateMPCScheduler(task_config);
      break;
#endif  // MPC_TASK_ENABLED
#ifdef PSI_TASK_ENABLED
    case rpc::TaskType::PSI_TASK:
      scheduler = std::make_unique<VMScheduler>();
      break;
#endif  // PSI_TASK_ENABLED
#ifdef PIR_TASK_ENABLED
    case rpc::TaskType::PIR_TASK:
      scheduler = std::make_unique<PIRScheduler>();
      break;
#endif  // PIR_TASK_ENABLED
    case rpc::TaskType::TEE_TASK:
      scheduler = std::make_unique<TEEScheduler>();
      break;
    default: {
      std::stringstream ss;
      ss << "Unsupported task type: " <<  rpc::TaskType_Name(task_type);
      RaiseException(ss.str());
    }
    }
    return scheduler;
  }
#ifdef MPC_TASK_ENABLED
  static std::unique_ptr<VMScheduler> CreateMPCScheduler(const rpc::Task& task_config) {
    std::unique_ptr<VMScheduler> scheduler{nullptr};
    auto& algorithm_type = task_config.code();
    if (algorithm_type == "maxpool") {
      scheduler = std::make_unique<CRYPTFLOW2Scheduler>();
    } else if (algorithm_type == "lenet") {
      scheduler = std::make_unique<FalconScheduler>();
    } else {
      scheduler = std::make_unique<ABY3Scheduler>();
    }
    return scheduler;
  }
#endif  // MPC_TASK_ENABLED
};

}  // namespace primihub::task
#endif  // SRC_PRIMIHUB_TASK_SEMANTIC_SCHEDULER_FACTORY_H_
