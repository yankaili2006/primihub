load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])

# Toggle for linking against @ypir//:ypir_matmul_kernels (upstream
# menonsamir/ypir src/matmul.cpp). When off (default),
# ypir_runtime.cc compiles into a stub returning retcode::FAIL with a
# "not vendored" message, and the @ypir dep is not pulled in.
# Turn on via:
#   bazel build --define=enable_ypir_real=1 \
#               --override_repository=ypir=<path>
config_setting(
    name = "enable_real",
    define_values = {"enable_ypir_real": "1"},
)


cc_library(
    name = "ypir_transpose",
    srcs = ["ypir_transpose.cc"],
    hdrs = ["ypir_transpose.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "ypir_matmul",
    srcs = ["ypir_matmul.cc"],
    hdrs = ["ypir_matmul.h"],
    defines = select({
        ":enable_real": ["PIR_YPIR_RUNTIME_VENDORED=1"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//src/primihub/common:common_defination",
    ] + select({
        ":enable_real": ["@ypir//:ypir_matmul_kernels"],
        "//conditions:default": [],
    }),
)


cc_library(
    name = "ypir_lwe_params",
    srcs = ["ypir_lwe_params.cc"],
    hdrs = ["ypir_lwe_params.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "ypir_negacyclic",
    srcs = ["ypir_negacyclic.cc"],
    hdrs = ["ypir_negacyclic.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "ypir_bits",
    srcs = ["ypir_bits.cc"],
    hdrs = ["ypir_bits.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "ypir_util",
    srcs = ["ypir_util.cc"],
    hdrs = ["ypir_util.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_operator",
    srcs = [
        "ypir.cc",
        "ypir_runtime.cc",
    ],
    hdrs = [
        "ypir.h",
        "ypir_runtime.h",
    ],
    defines = select({
        ":enable_real": ["PIR_YPIR_RUNTIME_VENDORED=1"],
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
        ":enable_real": ["@ypir//:ypir_matmul_kernels"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "ypir_modulus_switch",
    srcs = ["ypir_modulus_switch.cc"],
    hdrs = ["ypir_modulus_switch.h"],
    deps = [":ypir_bits"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_discrete_gaussian",
    srcs = ["ypir_discrete_gaussian.cc"],
    hdrs = ["ypir_discrete_gaussian.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_lwe",
    srcs = ["ypir_lwe.cc"],
    hdrs = ["ypir_lwe.h"],
    deps = [":ypir_negacyclic"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_chacha",
    srcs = ["ypir_chacha.cc"],
    hdrs = ["ypir_chacha.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_lwe_client",
    srcs = ["ypir_lwe_client.cc"],
    hdrs = ["ypir_lwe_client.h"],
    deps = [
        ":ypir_chacha",
        ":ypir_discrete_gaussian",
        ":ypir_lwe",
        ":ypir_lwe_params",
    ],
    visibility = ["//visibility:public"],
)
