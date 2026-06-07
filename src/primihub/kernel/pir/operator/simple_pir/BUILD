load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])

# Toggle for linking against @simplepir//:simplepir_c_kernels. When off
# (default), simple_pir_runtime.cc compiles into a stub returning
# retcode::FAIL with a "not vendored" message, and the @simplepir dep
# is not pulled in. Turn on via:
#   bazel build --define=enable_simple_pir_real=1 \
#               --override_repository=simplepir=<path>
config_setting(
    name = "enable_real",
    define_values = {"enable_simple_pir_real": "1"},
)

cc_library(
    name = "simple_pir_operator",
    srcs = [
        "simple_pir.cc",
        "simple_pir_runtime.cc",
    ],
    hdrs = [
        "simple_pir.h",
        "simple_pir_runtime.h",
    ],
    defines = select({
        ":enable_real": ["PIR_SIMPLE_PIR_RUNTIME_VENDORED=1"],
        "//conditions:default": [],
    }),
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ] + select({
        ":enable_real": ["@simplepir//:simplepir_c_kernels"],
        "//conditions:default": [],
    }),
)


# Algorithm-level static functions (Init / Setup) for SimplePIR. Built
# separately from simple_pir_operator so the operator can keep its
# current scaffolding flow while the protocol port lands in chunks.
# Setup needs Matrix::Mul, so the cc_library deps include pir_core::matrix
# which has its own enable_pir_core_real activation — keeps the two
# concerns orthogonal from the operator's enable_simple_pir_real flag.
cc_library(
    name = "simple_pir_protocol",
    srcs = ["simple_pir_protocol.cc"],
    hdrs = ["simple_pir_protocol.h"],
    deps = [
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator/pir_core:database",
        "//src/primihub/kernel/pir/operator/pir_core:lwe_params",
        "//src/primihub/kernel/pir/operator/pir_core:matrix",
        "@com_github_glog_glog//:glog",
    ],
)
