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
    name = "selector",
    hdrs = ["selector.h"],
    srcs = ["selector.cc"],
    deps = [
        ":capabilities",
        ":registry",
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
    name = "multi_peer_pir",
    hdrs = ["multi_peer_pir.h"],
    srcs = ["multi_peer_pir.cc"],
    deps = [
        ":base_pir_operator",
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/util/network:communication_lib",
        "@com_github_glog_glog//:glog",
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
    alwayslink = True,
)

cc_library(
    name = "keyword_pir_operator",
    deps = [
        "//src/primihub/kernel/pir/operator/keyword_pir_impl:keyword_pir_server_impl",
        "//src/primihub/kernel/pir/operator/keyword_pir_impl:keyword_pir_client_impl",
        "//src/primihub/kernel/pir/operator/keyword_pir_impl:keyword_pir_registrar",
    ],
)

# Aggregator (task 1.9 — single dep that pulls in registry + all
# default-enabled PIR algorithm operators so their alwayslink
# registrars fire at process start). Consumers (//:node,
# //src/primihub/cli/pir_inspect, future cc_binary targets) depend on
# :pir instead of enumerating the six operator targets locally — adding
# a new algorithm becomes a one-line change here. APSI / keyword_pir
# stays opt-in via select() on //:microsoft-apsi to match the historical
# build flag, since linking APSI pulls in microsoft-apsi's SEAL toolchain
# (cm/cpr/seal/zstd) and isn't free.
cc_library(
    name = "pir",
    deps = [
        ":registry",
        ":id_pir_operator",
        "//src/primihub/kernel/pir/operator/simple_pir:simple_pir_operator",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_operator",
        "//src/primihub/kernel/pir/operator/spiral_pir:spiral_pir_operator",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_pir_operator",
        "//src/primihub/kernel/pir/operator/tiptoe_pir:tiptoe_pir_operator",
    ] + select({
        # ypir hard-deps Intel HEXL (@hexl), which is x86/AVX512-only and
        # fails to compile on aarch64 (cpuinfo_x86.h). Drop it on arm.
        "//:aarch64": [],
        "//conditions:default": [
            "//src/primihub/kernel/pir/operator/ypir:ypir_operator",
        ],
    }) + select({
        "//:microsoft-apsi": [":keyword_pir_operator"],
        "//conditions:default": [],
    }),
)
