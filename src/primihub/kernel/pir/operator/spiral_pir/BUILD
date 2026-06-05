load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Phase-3 skeleton library. Registers SpiralPirOperator + capabilities so the
# Selector can see it; OnExecute is a clear-error stub until the real upstream
# crypto core lands (see openspec/changes/primihub-pir-multi-algo tasks
# 4.1, 4.2, 4.5, 4.7, 4.8 — all still open).
#
# alwayslink = True is mandatory: the registrar instance is the only thing in
# this .cc that the binary references, so without alwayslink the linker would
# strip the whole TU and the algorithm would silently not register.
cc_library(
    name = "spiral_pir_operator",
    srcs = ["spiral_pir.cc"],
    hdrs = ["spiral_pir.h"],
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ],
)
