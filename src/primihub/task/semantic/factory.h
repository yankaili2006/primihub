/*
 Copyright 2022 PrimiHub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

#ifndef SRC_PRIMIHUB_TASK_SEMANTIC_FACTORY_H_
#define SRC_PRIMIHUB_TASK_SEMANTIC_FACTORY_H_

#include <glog/logging.h>
#include <memory>

#include "src/primihub/task/semantic/task.h"
#ifdef MPC_TASK_ENABLED
#include "src/primihub/task/semantic/mpc_task.h"
#endif  // MPC_TASK_ENABLED

#ifdef PY_TASK_ENABLED
#include "src/primihub/task/semantic/fl_task.h"
#endif  // PY_TASK_ENABLED

#ifdef PSI_TASK_ENABLED
#include "src/primihub/task/semantic/psi_task.h"
#endif  // PSI_TASK_ENABLED

#ifdef PIR_TASK_ENABLED
#include "src/primihub/task/semantic/pir_task.h"
#endif  // PIR_TASK_ENABLED
#include "src/primihub/service/dataset/service.h"
#include "src/primihub/util/log.h"
#include "src/primihub/util/proto_log_helper.h"

using primihub::rpc::PushTaskRequest;
using primihub::rpc::Language;
using primihub::rpc::TaskType;
using primihub::service::DatasetService;
namespace pb_util = primihub::proto::util;
namespace primihub::task {

class TaskFactory {
 public:
  using RetType = std::pair<std::shared_ptr<TaskBase>, std::string>;
  static RetType Create(const std::string& node_id,
      const PushTaskRequest& request,
      std::shared_ptr<DatasetService> dataset_service,
      void* ra_service = nullptr,
      void* executor = nullptr) {
    auto task_language = request.task().language();
    const auto& task_info = request.task().task_info();
    std::string task_inof_str = pb_util::TaskInfoToString(task_info);
    auto task_type = request.task().type();
    RetType result_;
    switch (task_language) {
#ifdef PY_TASK_ENABLED
    case Language::PYTHON:
      result_ = std::make_pair(
          TaskFactory::CreateFLTask(node_id, request, dataset_service), "SUCCESS");
      break;
#endif
    case Language::PROTO: {
      switch (task_type) {
#ifdef MPC_TASK_ENABLED
      case rpc::TaskType::ACTOR_TASK:
        result_ = std::make_pair(
            TaskFactory::CreateMPCTask(node_id, request, dataset_service), "SUCCESS");
        break;
#endif  // MPC_TASK_ENABLED
#ifdef PSI_TASK_ENABLED
      case rpc::TaskType::PSI_TASK:
        result_ = std::make_pair(
            TaskFactory::CreatePSITask(node_id, request, dataset_service,
                                       ra_service, executor), "SUCCESS");
        break;
#endif // PSI_TASK_ENABLED
#ifdef PIR_TASK_ENABLED
      case rpc::TaskType::PIR_TASK:
        result_ = std::make_pair(
            TaskFactory::CreatePIRTask(node_id, request, dataset_service), "SUCCESS");;
        break;
#endif  // PIR_TASK_ENABLED
      default: {
        std::string err_msg = "Unsupported task type: " + rpc::TaskType_Name(task_type);
        LOG(ERROR) << task_inof_str << err_msg;
        result_ = std::make_pair(nullptr, err_msg);
        break;
      }
      }
      break;
    }
    default: {
      std::string err_msg = "Unsupported task type: " + rpc::Language_Name(task_language);
      LOG(ERROR) << task_inof_str << "unsupported language: " << task_language;
      result_ = std::make_pair(nullptr, err_msg);
      break;
    }
    }
    return result_;
  }

#ifdef PY_TASK_ENABLED
  static std::shared_ptr<TaskBase> CreateFLTask(const std::string& node_id,
      const PushTaskRequest& request,
      std::shared_ptr<DatasetService> dataset_service) {
    const auto& task_param = request.task();
    return std::make_shared<FLTask>(node_id, &task_param,
                                    request, dataset_service);
  }
#endif  // PY_TASK_ENABLED

#ifdef MPC_TASK_ENABLED
  static std::shared_ptr<TaskBase> CreateMPCTask(const std::string& node_id,
      const PushTaskRequest& request,
      std::shared_ptr<DatasetService> dataset_service) {
    const auto& _function_name = request.task().code();
    const auto& task_param = request.task();
    return std::make_shared<MPCTask>(node_id, _function_name,
                                    &task_param, dataset_service);
  }
#endif  // MPC_TASK_ENABLED

#ifdef PSI_TASK_ENABLED
  static std::shared_ptr<TaskBase> CreatePSITask(const std::string& node_id,
      const PushTaskRequest& request,
      std::shared_ptr<DatasetService> dataset_service,
      void* ra_server,
      void* tee_engine) {
    const auto& task_config = request.task();
    std::shared_ptr<TaskBase> task_ptr{nullptr};
    task_ptr = std::make_shared<PsiTask>(&task_config, dataset_service,
                                         ra_server, tee_engine);
    return task_ptr;
  }
#endif  // PSI_TASK_ENABLED

#ifdef PIR_TASK_ENABLED
  static std::shared_ptr<TaskBase> CreatePIRTask(const std::string& node_id,
      const PushTaskRequest& request,
      std::shared_ptr<DatasetService> dataset_service) {
    const auto& task_config = request.task();
    return std::make_shared<PirTask>(&task_config, dataset_service);
  }
#endif  // PIR_TASK_ENABLED

};
}  // namespace primihub::task

#endif  // SRC_PRIMIHUB_TASK_SEMANTIC_FACTORY_H_
