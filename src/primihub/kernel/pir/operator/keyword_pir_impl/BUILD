package(default_visibility = ["//visibility:public"])

C_OPTS = [
  "-w",
  "-D_ASPI",
]

DEP_OPTS = [
  ":common",
  "//src/primihub/kernel/pir/operator:base_pir_operator",
  "//src/primihub/util:endian_util",
  "//src/primihub/util:util_lib",
  "//src/primihub/protos:worker_proto",
  "@mircrosoft_apsi//:APSI",
]

cc_library(
  name = "keyword_pir_client_impl",
  hdrs = ["keyword_pir_client.h"],
  srcs = ["keyword_pir_client.cc"],
  copts = C_OPTS,
  deps = DEP_OPTS + [
    "@com_google_cityhash//:cityhash",
  ],
)

cc_library(
  name = "keyword_pir_server_impl",
  hdrs = ["keyword_pir_server.h"],
  srcs = ["keyword_pir_server.cc"],
  copts = C_OPTS,
  deps = DEP_OPTS,
)

cc_library(
  name = "common",
  hdrs = ["keyword_pir_common.h"],
)

# Self-registers KeywordPirOperator under name "apsi" via PirRegistrar.
# alwayslink = True is mandatory: registrar is the only externally-visible
# symbol in keyword_pir_registrar.cc, so without alwayslink the linker
# would strip the TU and the algorithm would silently not appear.
cc_library(
  name = "keyword_pir_registrar",
  srcs = ["keyword_pir_registrar.cc"],
  copts = C_OPTS,
  alwayslink = True,
  deps = [
    ":keyword_pir_client_impl",
    ":keyword_pir_server_impl",
    "//src/primihub/common:common_defination",
    "//src/primihub/kernel/pir:common_def",
    "//src/primihub/kernel/pir/operator:base_pir_operator",
    "//src/primihub/kernel/pir/operator:capabilities",
    "//src/primihub/kernel/pir/operator:registry",
    "@com_github_glog_glog//:glog",
  ],
)
