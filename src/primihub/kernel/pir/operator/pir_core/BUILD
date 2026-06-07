load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# Shared infra layer for the SimplePIR / DoublePIR / YPIR C++ ports.
# Provides primihub::pir::core::Matrix — uint32 row-major matrix that
# bridges shape-aware C++ to the @simplepir C matmul kernels established
# by tasks 5.1+5.2 and activated by 5.5 / 7.2 / 7.3 scaffolds.
#
# Same activation pattern as the per-operator runtime facades (see
# docs/pir/activation-pattern.md): the :enable_real config_setting
# injects both the PIR_PIR_CORE_REAL define and the @simplepir dep
# under --define=enable_pir_core_real=1. Stub mode compiles cleanly
# and kernel-calling methods return retcode::FAIL with the activation-
# flag guidance.
#
# alwayslink is NOT set here because Matrix has no self-registering
# globals — it is a normal cc_library that callers explicitly depend
# on. The operator BUILDs (simple_pir / double_pir / ypir) that opt
# into Matrix add this target to their deps; existing builds that do
# not use Matrix are unaffected.
config_setting(
    name = "enable_real",
    define_values = {"enable_pir_core_real": "1"},
)

cc_library(
    name = "matrix",
    srcs = ["matrix.cc"],
    hdrs = ["matrix.h"],
    defines = select({
        ":enable_real": ["PIR_PIR_CORE_REAL=1"],
        "//conditions:default": [],
    }),
    deps = [
        "//src/primihub/common:common_defination",
        "@com_github_glog_glog//:glog",
    ] + select({
        ":enable_real": ["@simplepir//:simplepir_c_kernels"],
        "//conditions:default": [],
    }),
)
