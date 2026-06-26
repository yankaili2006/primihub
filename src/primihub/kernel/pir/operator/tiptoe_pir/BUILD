load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

package(default_visibility = ["//visibility:public"])

# Tiptoe (Henzinger et al., SOSP'23) — BFV-on-SimplePIR semantic PIR.
# v1 skeleton (task 1.1a): registered operator + capability profile + dep edge.
# The real BFV-on-SimplePIR query path (vendored rlwe over SEAL + ported
# underhood LHE layer + reused pir::core SimplePIR) lands in chunks 1.1b-1.1f.
# See docs/pir/tiptoe-port-plan.md.
cc_library(
    name = "tiptoe_pir_operator",
    srcs = ["tiptoe_pir.cc"],
    hdrs = ["tiptoe_pir.h"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ],
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

# Toggle for linking the vendored @underhood//:rlwe BFV-on-SimplePIR crypto
# (over Microsoft SEAL). Off by default so the .50 build (no SEAL toolchain,
# GFW-blocked) stays green. Turn on for SEAL-available hosts via:
#   bazel test --define=enable_tiptoe_real=1 \
#              --override_repository=underhood=<path-to-ahenzinger/underhood>
config_setting(
    name = "enable_real",
    define_values = {"enable_tiptoe_real": "1"},
)

# RAII wrapper over the rlwe BFV context (chunk 1.1c). Header-only; the rlwe
# dep is real-mode only, so the header is compiled only by gated consumers.
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

# rlwe crypto-core smoke (chunk 1.1b). manual + gated: real mode needs SEAL;
# without the define it compiles to a single skipped test (no @underhood dep).
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

# rlwe Context RAII wrapper test (chunk 1.1c). manual + gated like rlwe_smoke.
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

# LHE client encrypt-secret roundtrip test (chunk 1.1d). manual + gated.
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

# LHE hint decompose/apply/recover test (chunk 1.1e). manual + gated.
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
