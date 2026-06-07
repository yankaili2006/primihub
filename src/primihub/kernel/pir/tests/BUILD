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

# Phase 3 / task 4.4 — SpiralRuntime end-to-end test. Behavior splits on
# PIR_SPIRAL_RUNTIME_VENDORED; the operator BUILD select() handles the
# define + dep wiring, so this target just inherits everything from the
# operator dep.
cc_test(
    name = "spiral_runtime_test",
    srcs = ["spiral_runtime_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/spiral_pir:spiral_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# P0 / task 1.8 — verifies the real APSI registrar fires when the binary is
# built with --define=microsoft-apsi=true. Built only under that flag,
# otherwise the keyword_pir_registrar cc_library is not part of the
# aggregator and the test would have no symbol to link against.
config_setting(
    name = "with_microsoft_apsi",
    define_values = {"microsoft-apsi": "true"},
)

cc_test(
    name = "keyword_pir_registrar_test",
    srcs = ["keyword_pir_registrar_test.cc"],
    target_compatible_with = select({
        ":with_microsoft_apsi": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator:factory",
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":with_microsoft_apsi": [
            "//src/primihub/kernel/pir/operator:keyword_pir_operator",
        ],
        "//conditions:default": [],
    }),
    size = "small",
)

# Phase 4 / task 5.5 scaffolding — DoublePirRuntime end-to-end smoke.
# Behaviour bifurcates on the operator BUILD select(); a single test
# file covers both modes via kDoublePirRuntimeVendored introspection.
cc_test(
    name = "double_pir_runtime_test",
    srcs = ["double_pir_runtime_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# Phase 6 / task 7.2 scaffolding — SimplePirRuntime end-to-end smoke.
cc_test(
    name = "simple_pir_runtime_test",
    srcs = ["simple_pir_runtime_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/simple_pir:simple_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# Phase 6 / task 7.3 scaffolding — YpirRuntime end-to-end smoke.
cc_test(
    name = "ypir_runtime_test",
    srcs = ["ypir_runtime_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/ypir:ypir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# pir_core::Matrix unit tests. Most tests are unconditional; the three
# kernel-bridge tests bifurcate on kPirCoreKernelsVendored. Build with
# --define=enable_pir_core_real=1 + --override_repository=simplepir=<p>
# to exercise the vendored branches.
cc_test(
    name = "matrix_test",
    srcs = ["matrix_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# LweParams unit tests — pure arithmetic, no kernel dependency.
cc_test(
    name = "lwe_params_test",
    srcs = ["lwe_params_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# GaussianSampler unit tests — pure arithmetic, statistical sanity.
cc_test(
    name = "gaussian_test",
    srcs = ["gaussian_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/pir_core:gaussian",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# Database unit tests — pure arithmetic on top of LweParams + Matrix.
# Runs the same in both stub and vendored modes (uses Matrix factories
# only; no kernel calls).
cc_test(
    name = "database_test",
    srcs = ["database_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# SimplePirProtocol::Init / Setup / GenSecret / Query tests. Init and
# GenSecret unconditional; Setup and Query bifurcate on
# core::kPirCoreKernelsVendored. Run with --define=enable_pir_core_real=1
# + --override_repository=simplepir=<p> to exercise the Matrix::Mul
# kernel path.
cc_test(
    name = "simple_pir_protocol_test",
    srcs = ["simple_pir_protocol_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:gaussian",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "//src/primihub/kernel/pir/operator/simple_pir:simple_pir_protocol",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)
