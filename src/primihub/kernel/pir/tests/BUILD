package(default_visibility = ["//visibility:public"])

cc_test(
    name = "registry_test",
    srcs = ["registry_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "capabilities_test",
    srcs = ["capabilities_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:capabilities",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "backend_test",
    srcs = ["backend_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/backend:backend",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "selector_test",
    srcs = ["selector_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator:selector",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "multi_peer_test",
    srcs = ["multi_peer_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:multi_peer_pir",
        "//src/primihub/util/network:communication_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)
