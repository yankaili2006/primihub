package(default_visibility = ["//visibility:public"])

cc_library(
    name = "factory",
    hdrs = ["factory.h"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        ":base_pir_operator",
        ":registry",
        ":id_pir_operator",
    ] + select({
        "//:microsoft-apsi": [":keyword_pir_operator"],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "capabilities",
    hdrs = ["capabilities.h"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
    ],
)

cc_library(
    name = "registry",
    hdrs = ["registry.h"],
    srcs = ["registry.cc"],
    deps = [
        ":base_pir_operator",
        ":capabilities",
        "//src/primihub/kernel/pir:common_def",
        "@com_github_glog_glog//:glog",
    ],
)

cc_library(
    name = "base_pir_operator",
    hdrs = ["base_pir.h"],
    srcs = ["base_pir.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/util:endian_util",
        "//src/primihub/util:util_lib",
        "//src/primihub/common:common_defination",
        "//src/primihub/util/network:communication_lib",
    ],
)

cc_library(
    name = "id_pir_operator",
    hdrs = ["id_pir.h"],
    srcs = ["id_pir.cc"],
    deps = [
        ":base_pir_operator",
        ":registry",
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/util/network:communication_lib",
    ],
    alwayslink = True,  # static registrar must not be DCE'd
)

cc_library(
    name = "keyword_pir_operator",
    deps = [
        "//src/primihub/kernel/pir/operator/keyword_pir_impl:keyword_pir_server_impl",
        "//src/primihub/kernel/pir/operator/keyword_pir_impl:keyword_pir_client_impl",
    ],
)
