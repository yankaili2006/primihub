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
        "@com_github_base64_cpp//:base64_lib",
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

cc_library(
    name = "ypir_convolution",
    srcs = ["ypir_convolution.cc"],
    hdrs = ["ypir_convolution.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_convolution_ntt",
    srcs = ["ypir_convolution_ntt.cc"],
    hdrs = ["ypir_convolution_ntt.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@hexl//:hexl",
        ":ypir_arith",
    ],
)

cc_library(
    name = "ypir_arith",
    srcs = ["ypir_arith.cc"],
    hdrs = ["ypir_arith.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_params",
    srcs = ["ypir_params.cc"],
    hdrs = ["ypir_params.h"],
    visibility = ["//visibility:public"],
    deps = [":ypir_arith"],
)

cc_library(
    name = "ypir_poly",
    srcs = ["ypir_poly.cc"],
    hdrs = ["ypir_poly.h"],
    visibility = ["//visibility:public"],
    deps = [":ypir_arith", ":ypir_params", ":ypir_poly_types", "@hexl//:hexl"],
)

cc_library(
    name = "ypir_poly_types",
    hdrs = ["ypir_poly_types.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ypir_poly_ops",
    srcs = ["ypir_poly_ops.cc"],
    hdrs = ["ypir_poly_ops.h"],
    visibility = ["//visibility:public"],
    deps = [":ypir_arith", ":ypir_params", ":ypir_poly_types"],
)

cc_library(
    name = "ypir_gadget",
    srcs = ["ypir_gadget.cc"],
    hdrs = ["ypir_gadget.h"],
    visibility = ["//visibility:public"],
    deps = [":ypir_params", ":ypir_poly_types"],
)

cc_library(
    name = "ypir_packing_fast",
    srcs = ["ypir_packing_fast.cc"],
    hdrs = ["ypir_packing_fast.h"],
    visibility = ["//visibility:public"],
    deps = [":ypir_arith", ":ypir_params", ":ypir_poly_types"],
)

cc_library(
    name = "ypir_regev",
    srcs = ["ypir_regev.cc"],
    hdrs = ["ypir_regev.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_chacha",
        ":ypir_discrete_gaussian",
        ":ypir_gadget",
        ":ypir_modulus_switch",
        ":ypir_params",
        ":ypir_poly",
        ":ypir_poly_ops",
        ":ypir_poly_types",
    ],
)

cc_library(
    name = "ypir_server",
    srcs = ["ypir_server.cc"],
    hdrs = ["ypir_server.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_bits",
        ":ypir_lwe_params",
        ":ypir_poly_ops",
        ":ypir_transpose",
        ":ypir_util",
        ":ypir_params",
        ":ypir_poly",
        ":ypir_poly_types",
            ":ypir_arith",
        ":ypir_chacha",
        ":ypir_convolution_ntt",
        ":ypir_negacyclic",
        ":ypir_scheme",
        ":ypir_client",
        ":ypir_spiral_client",
        ":ypir_kernel",
],
)

cc_library(
    name = "ypir_packing",
    srcs = ["ypir_packing.cc"],
    hdrs = ["ypir_packing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_arith",
        ":ypir_util",
        ":ypir_modulus_switch",
        ":ypir_params",
        ":ypir_poly",
        ":ypir_poly_ops",
        ":ypir_poly_types",
        ":ypir_regev",
        ":ypir_server",
    ],
)

cc_library(
    name = "ypir_scheme",
    srcs = ["ypir_scheme.cc"],
    hdrs = ["ypir_scheme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_params",
    ],
)

cc_library(
    name = "ypir_client",
    srcs = ["ypir_client.cc"],
    hdrs = ["ypir_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_chacha",
        ":ypir_lwe_client",
        ":ypir_lwe_params",
        ":ypir_modulus_switch",
        ":ypir_negacyclic",
        ":ypir_params",
        ":ypir_poly",
        ":ypir_poly_types",
        ":ypir_scheme",
        ":ypir_spiral_client",
    ],
)

cc_library(
    name = "ypir_spiral_client",
    srcs = ["ypir_spiral_client.cc"],
    hdrs = ["ypir_spiral_client.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_chacha",
        ":ypir_discrete_gaussian",
        ":ypir_params",
        ":ypir_poly",
        ":ypir_poly_ops",
        ":ypir_poly_types",
        ":ypir_regev",
    ],
)

cc_library(
    name = "ypir_kernel",
    srcs = ["ypir_kernel.cc"],
    hdrs = ["ypir_kernel.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ypir_arith",
        ":ypir_params",
    ],
)
