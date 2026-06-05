load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Phase-4 skeleton library for DoublePIR (USENIX'23).
# Mirrors the spiral_pir_operator pattern: registers the algorithm + caps
# at static-init time via the AddPending → EnsureRegistered two-phase
# path (see commit 5d86cafc), while OnExecute returns a loud FAIL until
# the real LWE-based query path lands (tasks 5.4-5.10).
#
# alwayslink = True is mandatory — without it the linker drops the TU
# because the binary never names DoublePirOperator directly.
cc_library(
    name = "double_pir_operator",
    srcs = ["double_pir.cc"],
    hdrs = ["double_pir.h"],
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:multi_peer_pir",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ],
)
