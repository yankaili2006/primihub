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

# HintCache unit tests — fingerprint stability + LRU + GetOrComputeHint hit/miss.
cc_test(
    name = "hint_cache_test",
    srcs = ["hint_cache_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/double_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# HintGen unit tests — null guards + shape contract + per-stage timing.
cc_test(
    name = "hint_gen_test",
    srcs = ["hint_gen_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/double_pir:hint_gen",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
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
        "//src/primihub/kernel/pir/operator/double_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
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

# task 7.2 final wiring — SimplePirOperator OnExecute integration test.
# Drives the full SimplePirProtocol pipeline through the operator's
# PirDataType interface; bifurcates on core::kPirCoreKernelsVendored.
cc_test(
    name = "simple_pir_operator_test",
    srcs = ["simple_pir_operator_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
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

cc_test(
    name = "ypir_transpose_test",
    srcs = ["ypir_transpose_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_transpose",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_matmul_test",
    srcs = ["ypir_matmul_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_matmul",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_lwe_params_test",
    srcs = ["ypir_lwe_params_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_negacyclic_test",
    srcs = ["ypir_negacyclic_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_negacyclic",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_bits_test",
    srcs = ["ypir_bits_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_bits",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_util_test",
    srcs = ["ypir_util_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_negacyclic",
        "//src/primihub/kernel/pir/operator/ypir:ypir_util",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_lwe_consts_test",
    srcs = ["frodo_lwe_consts_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_lwe_consts",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_format_test",
    srcs = ["frodo_format_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_format",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_prng_test",
    srcs = ["frodo_prng_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_prng",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_matrices_test",
    srcs = ["frodo_matrices_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_matrices",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_params_test",
    srcs = ["frodo_params_test.cc"],
    deps = [
        ":frodo_test_helpers",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_database",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_matrices",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_params",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_prng",
        "@com_github_base64_cpp//:base64_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_pir_test",
    srcs = ["frodo_pir_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_pir_operator",
        "@com_github_base64_cpp//:base64_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_api_test",
    srcs = ["frodo_api_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_api",
        "@com_github_base64_cpp//:base64_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_database_test",
    srcs = ["frodo_database_test.cc"],
    deps = [
        ":frodo_test_helpers",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_database",
        "@com_github_base64_cpp//:base64_lib",
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

cc_test(
    name = "double_pir_protocol_test",
    srcs = ["double_pir_protocol_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_protocol",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# HintSerialize unit tests — round-trip, magic/version/reserved guards,
# truncated and trailing-byte rejection. Task 5.6 chunk 3.
cc_test(
    name = "hint_serialize_test",
    srcs = ["hint_serialize_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/double_pir:hint_serialize",
        "//src/primihub/kernel/pir/operator/double_pir:hint_gen",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# SimpleHintCache unit tests — fingerprint + LRU + GetOrComputeHint.
# Mirrors hint_cache_test for the SimplePIR sibling implementation.
cc_test(
    name = "simple_pir_hint_cache_test",
    srcs = ["simple_pir_hint_cache_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/simple_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

# SimplePIR serialize + persistence + operator-integration tests
# (chunks 3+4+5 sibling of DoublePIR's coverage).
cc_test(
    name = "simple_pir_serialize_persist_test",
    srcs = ["simple_pir_serialize_persist_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "//src/primihub/kernel/pir/operator/simple_pir:simple_pir_operator",
        "//src/primihub/kernel/pir/operator/simple_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/simple_pir:hint_serialize",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "double_pir_hint_link_test",
    srcs = ["double_pir_hint_link_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/double_pir:hint_gen",
        "//src/primihub/kernel/pir/operator/double_pir:hint_link",
        "//src/primihub/kernel/pir/operator/double_pir:hint_serialize",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "//src/primihub/util/network:communication_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "double_pir_role_test",
    srcs = ["double_pir_role_test.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_operator",
        "//src/primihub/kernel/pir/operator/double_pir:hint_cache",
        "//src/primihub/kernel/pir/operator/double_pir:hint_gen",
        "//src/primihub/kernel/pir/operator/double_pir:hint_link",
        "//src/primihub/kernel/pir/operator/double_pir:hint_serialize",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "//src/primihub/util/network:communication_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "frodo_flat_matrix_test",
    srcs = ["frodo_flat_matrix_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_flat_matrix",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_library(
    name = "frodo_test_helpers",
    testonly = True,
    hdrs = ["frodo_test_helpers.h"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_flat_matrix",
    ],
    visibility = ["//visibility:private"],
)

cc_test(
    name = "ypir_modulus_switch_test",
    srcs = ["ypir_modulus_switch_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_discrete_gaussian_test",
    srcs = ["ypir_discrete_gaussian_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_discrete_gaussian",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_lwe_test",
    srcs = ["ypir_lwe_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_chacha_test",
    srcs = ["ypir_chacha_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_lwe_client_test",
    srcs = ["ypir_lwe_client_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_spiral_smoke_test",
    srcs = ["ypir_spiral_smoke_test.cc"],
    tags = ["manual"],
    deps = [
        "@com_google_googletest//:gtest_main",
        "@spiral_pir//:spiral_lib",
    ],
    size = "small",
)

cc_test(
    name = "ypir_convolution_test",
    srcs = ["ypir_convolution_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_convolution",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_convolution_ntt_test",
    srcs = ["ypir_convolution_ntt_test.cc"],
    tags = ["manual"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_convolution",
        "//src/primihub/kernel/pir/operator/ypir:ypir_convolution_ntt",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_arith_test",
    srcs = ["ypir_arith_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_arith",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_params_test",
    srcs = ["ypir_params_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_arith",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_poly_test",
    srcs = ["ypir_poly_test.cc"],
    tags = ["manual"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_poly_ops_test",
    srcs = ["ypir_poly_ops_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_ops",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_gadget_test",
    srcs = ["ypir_gadget_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_gadget",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_packing_fast_test",
    srcs = ["ypir_packing_fast_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_arith",
        "//src/primihub/kernel/pir/operator/ypir:ypir_packing_fast",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_ops",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_regev_test",
    srcs = ["ypir_regev_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_discrete_gaussian",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_ops",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "//src/primihub/kernel/pir/operator/ypir:ypir_gadget",
        "//src/primihub/kernel/pir/operator/ypir:ypir_regev",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_server_test",
    srcs = ["ypir_server_test.cc"],
    tags = ["manual"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_packing_test",
    srcs = ["ypir_packing_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_arith",
        "//src/primihub/kernel/pir/operator/ypir:ypir_util",
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_discrete_gaussian",
        "//src/primihub/kernel/pir/operator/ypir:ypir_gadget",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_packing",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_ops",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "//src/primihub/kernel/pir/operator/ypir:ypir_regev",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_server_yserver_test",
    srcs = ["ypir_server_yserver_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_server_ring_test",
    srcs = ["ypir_server_ring_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_ops",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "//src/primihub/kernel/pir/operator/ypir:ypir_transpose",
        "//src/primihub/kernel/pir/operator/ypir:ypir_util",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_scheme_test",
    srcs = ["ypir_scheme_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_scheme",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_client_test",
    srcs = ["ypir_client_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_negacyclic",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_spiral_client_test",
    srcs = ["ypir_spiral_client_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "//src/primihub/kernel/pir/operator/ypir:ypir_spiral_client",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_yclient_test",
    srcs = ["ypir_yclient_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_lwe_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly_types",
        "//src/primihub/kernel/pir/operator/ypir:ypir_scheme",
        "//src/primihub/kernel/pir/operator/ypir:ypir_spiral_client",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_server_hint_test",
    srcs = ["ypir_server_hint_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_arith",
        "//src/primihub/kernel/pir/operator/ypir:ypir_kernel",
        "//src/primihub/kernel/pir/operator/ypir:ypir_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_spiral_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_convolution",
        "//src/primihub/kernel/pir/operator/ypir:ypir_negacyclic",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_scheme",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_kernel_test",
    srcs = ["ypir_kernel_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_kernel",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_e2e_test",
    srcs = ["ypir_e2e_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_discrete_gaussian",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_packing",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_regev",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "//src/primihub/kernel/pir/operator/ypir:ypir_spiral_client",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_operator_test",
    srcs = ["ypir_operator_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator/ypir:ypir_operator",
        "@com_github_base64_cpp//:base64_lib",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)

cc_test(
    name = "ypir_e2e_large_test",
    srcs = ["ypir_e2e_large_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/ypir:ypir_chacha",
        "//src/primihub/kernel/pir/operator/ypir:ypir_client",
        "//src/primihub/kernel/pir/operator/ypir:ypir_discrete_gaussian",
        "//src/primihub/kernel/pir/operator/ypir:ypir_modulus_switch",
        "//src/primihub/kernel/pir/operator/ypir:ypir_packing",
        "//src/primihub/kernel/pir/operator/ypir:ypir_params",
        "//src/primihub/kernel/pir/operator/ypir:ypir_poly",
        "//src/primihub/kernel/pir/operator/ypir:ypir_regev",
        "//src/primihub/kernel/pir/operator/ypir:ypir_server",
        "//src/primihub/kernel/pir/operator/ypir:ypir_spiral_client",
        "@com_google_googletest//:gtest_main",
    ],
    size = "medium",
)

cc_test(
    name = "tiptoe_pir_test",
    srcs = ["tiptoe_pir_test.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator/tiptoe_pir:tiptoe_pir_operator",
        "@com_google_googletest//:gtest_main",
    ],
    size = "small",
)
