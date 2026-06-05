package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "pir_inspect",
    srcs = ["pir_inspect_main.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator:capabilities",
        "//src/primihub/kernel/pir/operator:registry",
        "//src/primihub/kernel/pir/operator:selector",
        "//src/primihub/kernel/pir/operator:id_pir_operator",
        "//src/primihub/kernel/pir/operator/spiral_pir:spiral_pir_operator",
        "@com_github_glog_glog//:glog",
        # The legacy id_pir operator is alwayslink so its registrar runs
        # at startup. Future algorithms (spiral / double / etc) get pulled
        # in here as deps when implemented; until then list() shows only
        # the algorithms currently linked in.
    ],
)
