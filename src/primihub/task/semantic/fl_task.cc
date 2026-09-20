/* Copyright 2022 PrimiHub

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

#include "src/primihub/task/semantic/fl_task.h"
#include <glog/logging.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>
#include "src/primihub/util/util.h"
#include "base64.h"
#include <google/protobuf/text_format.h>
#include "pybind11/embed.h"
#include <mutex>

namespace primihub::task {
using Process = Poco::Process;
using ProcessHandle = Poco::ProcessHandle;
namespace py = pybind11;
FLTask::FLTask(const std::string& node_id,
               const TaskParam* task_param,
               const PushTaskRequest& task_request,
               std::shared_ptr<DatasetService> dataset_service) :
               TaskBase(task_param, dataset_service),
               task_request_(&task_request) {}

int FLTask::execute() {
  std::string pb_task_request_;
  bool succ_flag = task_request_->SerializeToString(&pb_task_request_);
  if (!succ_flag) {
    LOG(ERROR) << "ill formatted task request";
    return -1;
  }

  // Persistent, process-wide interpreter: initialize once and NEVER finalize.
  // Per-task Py_Finalize/Py_Initialize (scoped_interpreter) corrupts C-extensions
  // that don't support re-init (grpc cygrpc -> CompressionAlgorithm None ->
  // AttributeError on 2nd+ task). Keeping one interpreter reuses sys.modules.
  static std::once_flag ph_py_init_flag;
  std::call_once(ph_py_init_flag, []() {
    py::initialize_interpreter();
    PyEval_SaveThread();  // release GIL so worker threads can acquire it
  });
  py::gil_scoped_acquire ph_gil;
  try {
  VLOG(1) << "<<<<<<<<< Import PrmimiHub Python Executor <<<<<<<<<";
  py::object ph_exec_m_ =
      py::module::import("primihub.executor").attr("Executor");
  py::object ph_context_m_ = py::module::import("primihub.context");
  py::object set_message;
  set_message = ph_context_m_.attr("set_message");
  set_message(py::bytes(pb_task_request_));
  set_message.release();
  auto& server_config = primihub::ServerConfig::getInstance();
  auto& host_cfg = server_config.getServiceConfig();
  if (host_cfg.use_tls()) {
    auto& cert_config = server_config.getCertificateConfig();
    auto root_ca_path = cert_config.rootCAPath();
    auto key_path = cert_config.keyPath();
    auto cert_path = cert_config.certPath();
    VLOG(1) << "Set cert config info, root_ca_path: " << root_ca_path << " "
        << "key_path: " << key_path << " "
        << "cert_path: " << cert_path;
    py::object set_cert_config;
    set_cert_config = ph_context_m_.attr("set_cert_config");
    set_cert_config(root_ca_path, key_path, cert_path);
    set_cert_config.release();
  }
  VLOG(1) << "<<<<<<<<< Start executing Python code <<<<<<<<<" << std::endl;
  // Execute python code. Catch python exceptions WHILE the interpreter is still
  // alive; otherwise error_already_set's dtor runs after scoped_interpreter is
  // finalized and PyGILState_Ensure() derefs a null semaphore -> segfault.
  ph_exec_m_.attr("execute_py")();
  VLOG(1) << "<<<<<<<<< Execute Python Code End <<<<<<<<<" << std::endl;
  py::object mpc_util = py::module::import("primihub.MPC.util");
  py::object stop_aux_task = mpc_util.attr("stop_auxiliary_party");
  stop_aux_task();
  stop_aux_task.release();
  VLOG(1) << "<<<<<<<<< Clean Task Env End <<<<<<<<<" << std::endl;
  } catch (py::error_already_set& e) {
    LOG(ERROR) << "FL python execution error: " << e.what();
    return -1;
  } catch (const std::exception& e) {
    LOG(ERROR) << "FL execution std error: " << e.what();
    return -1;
  }
  return 0;
}

}  // namespace primihub::task
