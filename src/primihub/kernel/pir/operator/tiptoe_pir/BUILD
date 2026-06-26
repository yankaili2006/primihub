load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

# Toggle for linking the vendored @underhood//:rlwe BFV-on-SimplePIR crypto
# (over Microsoft SEAL). Off by default so the .50 build (no SEAL toolchain,
# GFW-blocked) stays green. Turn on for SEAL-available hosts via:
#   bazel test --define=enable_tiptoe_real=1 \
#              --override_repository=underhood=<path-to-ahenzinger/underhood>
config_setting(
    name = "enable_real",
    define_values = {"enable_tiptoe_real": "1"},
)

# Tiptoe (Henzinger et al., SOSP'23) — BFV-on-SimplePIR semantic PIR (task 1.1).
# Default build: skeleton (OnExecute -> FAIL, is_real=false). With
# --define=enable_tiptoe_real=1: real LHE-on-SimplePIR retrieval via
# tiptoe_lhe_pir (validated standalone vs SEAL 4.1). See
# docs/pir/tiptoe-port-plan.md.
cc_library(
    name = "tiptoe_pir_operator",
    srcs = ["tiptoe_pir.cc"],
    hdrs = ["tiptoe_pir.h"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ] + select({
        ":enable_real": [
            ":tiptoe_lhe_pir",
            "@com_github_base64_cpp//:base64_lib",
        ],
        "//conditions:default": [],
    }),
    alwayslink = True,
)

# Pure secret-key helpers (chunk 1.1c) — zero-dep, builds + tests on .50.
cc_library(
    name = "tiptoe_secret",
    hdrs = ["tiptoe_secret.h"],
)

# Pure limb-decomposition helpers (chunk 1.1e) — zero-dep, builds + tests on .50.
cc_library(
    name = "tiptoe_limb",
    hdrs = ["tiptoe_limb.h"],
)

# RAII wrapper over the rlwe BFV context (chunk 1.1c). Header-only; real-mode.
cc_library(
    name = "tiptoe_params",
    hdrs = ["tiptoe_params.h"],
    deps = select({
        ":enable_real": ["@underhood//:rlwe"],
        "//conditions:default": [],
    }),
)

# LHE client encrypt-secret core (chunk 1.1d). Header-only; real-mode only.
cc_library(
    name = "tiptoe_client",
    hdrs = ["tiptoe_client.h"],
    deps = [
        ":tiptoe_params",
        ":tiptoe_secret",
    ],
)

# LHE hint decompose/apply/recover (chunk 1.1e). Header-only; real-mode only.
cc_library(
    name = "tiptoe_hint",
    hdrs = ["tiptoe_hint.h"],
    deps = [
        ":tiptoe_client",
        ":tiptoe_limb",
        ":tiptoe_params",
        ":tiptoe_secret",
    ],
)

# Self-contained LHE-on-SimplePIR retrieval (chunk 1.1f). Header-only; real-mode.
cc_library(
    name = "tiptoe_lhe_pir",
    hdrs = ["tiptoe_lhe_pir.h"],
    deps = [
        ":tiptoe_client",
        ":tiptoe_hint",
        ":tiptoe_params",
    ],
)

# --- gated manual crypto/unit tests (real mode needs SEAL) ---

cc_test(
    name = "rlwe_smoke_test",
    srcs = ["rlwe_smoke_test.cc"],
    tags = ["manual"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":enable_real": ["@underhood//:rlwe"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "tiptoe_params_test",
    srcs = ["tiptoe_params_test.cc"],
    tags = ["manual"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":enable_real": [":tiptoe_params"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "tiptoe_client_test",
    srcs = ["tiptoe_client_test.cc"],
    tags = ["manual"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":enable_real": [":tiptoe_client"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "tiptoe_hint_test",
    srcs = ["tiptoe_hint_test.cc"],
    tags = ["manual"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":enable_real": [":tiptoe_hint"],
        "//conditions:default": [],
    }),
)

# End-to-end LHE-on-SimplePIR retrieval test (chunk 1.1f). manual + gated.
cc_test(
    name = "tiptoe_lhe_pir_test",
    srcs = ["tiptoe_lhe_pir_test.cc"],
    tags = ["manual"],
    defines = select({
        ":enable_real": ["PIR_TIPTOE_RLWE_VENDORED=1"],
        "//conditions:default": [],
    }),
    deps = [
        "@com_google_googletest//:gtest_main",
    ] + select({
        ":enable_real": [":tiptoe_lhe_pir"],
        "//conditions:default": [],
    }),
)
