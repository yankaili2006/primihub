load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])


cc_library(
    name = "frodo_lwe_consts",
    hdrs = ["frodo_lwe_consts.h"],
    visibility = ["//visibility:public"],
)


cc_library(
    name = "frodo_format",
    srcs = ["frodo_format.cc"],
    hdrs = ["frodo_format.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//src/primihub/common:common_defination",
        "@com_github_base64_cpp//:base64_lib",
    ],
)





cc_library(
    name = "frodo_prng",
    srcs = ["frodo_prng.cc"],
    hdrs = ["frodo_prng.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "frodo_matrices",
    srcs = ["frodo_matrices.cc"],
    hdrs = ["frodo_matrices.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":frodo_prng",
        "//src/primihub/common:common_defination",
    ],
)



cc_library(
    name = "frodo_database",
    srcs = ["frodo_database.cc"],
    hdrs = ["frodo_database.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":frodo_format",
        ":frodo_matrices",
        "//src/primihub/common:common_defination",
        "@com_github_base64_cpp//:base64_lib",
    ],
)

cc_library(
    name = "frodo_pir_operator",
    srcs = ["frodo_pir.cc"],
    hdrs = ["frodo_pir.h"],
    alwayslink = True,
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:base_pir_operator",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "@com_github_glog_glog//:glog",
    ],
)
