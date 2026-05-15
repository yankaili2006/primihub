cc_test(
    name = "share_test",
    srcs = [
        "aby3/share_test.cc",
    ],
    deps = [
        "//test/primihub:test_util_test",
        "//src/primihub/common:common_lib",
        "//src/primihub/util/network:network_lib",
        "//src/primihub/protocol:share_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_glog_glog//:glog",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
cc_test(
    name = "protocol_aby3_test",
    srcs = [
      "aby3/evaluator/binary_evaluator_test.cc",
    ],
    deps = [
        "//src/primihub/protocol:protocol_aby3_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "protocol_aby3_encryptor_test",
    srcs = [
      "aby3/encryptor_test.cc",
    ],
    deps = [
        "//src/primihub/protocol:protocol_aby3_lib",
        "//src/primihub/util:util_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "protocol_aby3_runtime_test",
    srcs = [
      "aby3/runtime_test.cc",
    ],
    deps = [
        "//src/primihub/protocol:protocol_aby3_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "protocol_aby3_sh3_gen_test",
    srcs = [
      "aby3/sh3_gen_test.cc",
    ],
    deps = [
        "//src/primihub/protocol:protocol_aby3_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "psi_protocol_test",
    srcs = [
      "psi_test.cc",
    ],
    deps = [
        "//src/primihub/kernel/psi:psi_kernel",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

cc_test(
    name = "pir_protocol_test",
    srcs = [
      "pir_test.cc",
    ],
    deps = [
        "//src/primihub/kernel/pir:pir_kernel",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)