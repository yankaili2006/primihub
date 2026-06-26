load("@rules_cc//cc:defs.bzl", "cc_library")

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
