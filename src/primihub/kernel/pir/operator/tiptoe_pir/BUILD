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

# Toggle for linking the vendored @underhood//:rlwe BFV-on-SimplePIR crypto
# (over Microsoft SEAL). Off by default so the .50 build (no SEAL toolchain,
# GFW-blocked) stays green. Turn on for SEAL-available hosts via:
#   bazel test --define=enable_tiptoe_real=1 \
#              --override_repository=underhood=<path-to-ahenzinger/underhood>
config_setting(
    name = "enable_real",
    define_values = {"enable_tiptoe_real": "1"},
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
