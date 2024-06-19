package(default_visibility = ["//visibility:public"])
config_setting(
  name = "py_task_enabled",
  values = {"define" : "enable_py_task=true"},
)
config_setting(
  name = "mpc_task_enabled",
  values = {"define" : "enable_mpc_task=true"},
)
config_setting(
  name = "pir_task_enabled",
  values = {"define" : "enable_pir_task=true"},
)
config_setting(
  name = "psi_task_enabled",
  values = {"define" : "enable_psi_task=true"},
)

SCHEDULER_DEFINES_OPTS = [
  ] + select({
    "py_task_enabled": ["PY_TASK_ENABLED"],
    "//conditions:default": []
  }) + select({
    "mpc_task_enabled": ["MPC_TASK_ENABLED"],
    "//conditions:default": []
  }) + select({
    "pir_task_enabled": ["PIR_TASK_ENABLED"],
    "//conditions:default": []
  }) + select({
    "psi_task_enabled": ["PSI_TASK_ENABLED"],
    "//conditions:default": []
  })

SCHEDULER_DEPS_OPTS = [
    ":tee_scheduler",
  ] + select({
    "py_task_enabled": [":fl_scheduler"],
    "//conditions:default": []
  }) + select({
    "mpc_task_enabled": [
      ":mpc_scheduler",
      ":aby3_scheduler",
    ],
    "//conditions:default": []
  }) + select({
    "pir_task_enabled": [":pir_scheduler"],
    "//conditions:default": []
  }) + select({
    "psi_task_enabled": [":scheduler_interface"],
    "//conditions:default": []
  })

cc_library(
  name = "scheduler_factory",
  srcs = ["factory.h"],
  defines = SCHEDULER_DEFINES_OPTS,
  deps = [
    ":scheduler_lib",
  ],
)

cc_library(
  name = "scheduler_interface",
  hdrs = [
    "scheduler.h",
  ],
  srcs = [
    "scheduler.cc"
  ],
  deps = [
    "//src/primihub/common/config:server_config",
    "//src/primihub/protos:worker_proto",
    "//src/primihub/util/network:communication_lib",
    "//src/primihub/common:common_defination",
    "//src/primihub/service:dataset_service",
    "@com_github_glog_glog//:glog",
  ],
)

cc_library(
  name = "scheduler_lib",
  deps = SCHEDULER_DEPS_OPTS,
)

SCHEDULER_DEPS = [
  ":scheduler_interface",
  "//src/primihub/protos:common_proto",
]

cc_library(
    name = "mpc_scheduler",
    hdrs = ["mpc_scheduler.h"],
    srcs = ["mpc_scheduler.cc"],
    deps = SCHEDULER_DEPS,
)

cc_library(
    name = "aby3_scheduler",
    hdrs = ["aby3_scheduler.h"],
    srcs = ["aby3_scheduler.cc"],
    deps = SCHEDULER_DEPS,
)

cc_library(
    name = "fl_scheduler",
    hdrs = ["fl_scheduler.h"],
    srcs = ["fl_scheduler.cc"],
    deps = SCHEDULER_DEPS + [
        "//src/primihub/task:task_common",
        "//src/primihub/task/language:python_parser",
    ],
)

cc_library(
    name = "pir_scheduler",
    hdrs = ["pir_scheduler.h"],
    srcs = ["pir_scheduler.cc"],
    deps = SCHEDULER_DEPS,
)

cc_library(
    name = "psi_scheduler",
    hdrs = ["psi_scheduler.h"],
    srcs = ["psi_scheduler.cc"],
    deps = SCHEDULER_DEPS,
)

cc_library(
    name = "tee_scheduler",
    hdrs = ["tee_scheduler.h"],
    srcs = ["tee_scheduler.cc"],
    deps = SCHEDULER_DEPS,
)