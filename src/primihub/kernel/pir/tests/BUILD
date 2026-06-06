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

cc_test(
    name = "spiral_pir_test",
    srcs = ["spiral_pir_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator/spiral_pir:spiral_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "double_pir_test",
    srcs = ["double_pir_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator:selector",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# P5 — verifies legacy PirType wire-format still routes through the new
# registry-based Factory shim. Pulls in the factory + id_pir_operator so
# the real legacy ID_PIR registrar runs and the back-compat path is end-
# to-end exercised; the KEY_PIR / "apsi" leg falls back to a stub when
# APSI isn't linked into this build.
cc_test(
    name = "proto_compat_test",
    srcs = ["proto_compat_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator:factory",
        "//src/primihub/kernel/pir/operator:id_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# P5 — Selector behaviours that pir_inspect `auto` command exercises:
# query-type filtering, two-server opt-in gate, latency-budget cutoff,
# Recommend vs RecommendWithRationale shape. Uses stub algorithms with
# known capability profiles so assertions don't depend on which real
# algorithms happen to be linked.
cc_test(
    name = "cli_test",
    srcs = ["cli_test.cc"],
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

# Phase 3 / task 4.5 — EstimateParams unit tests. Header-only target, so the
# only dep is the operator (which exports params.h via its hdrs list).
cc_test(
    name = "spiral_params_test",
    srcs = ["spiral_params_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/spiral_pir:spiral_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)
