load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Standalone PIR operator benchmark (not a gtest): times OnExecute for
# pirana vs frodo_pir vs id_pir over an identical logical DB. Run on the
# bench host:
#   bazel run --config=linux_x86_64 //src/primihub/kernel/pir/bench:pir_operator_bench
cc_binary(
    name = "pir_operator_bench",
    srcs = ["pir_operator_bench.cc"],
    deps = [
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_pir_operator",
        "//src/primihub/kernel/pir/operator:id_pir_operator",
        "//src/primihub/kernel/pir/operator/pirana_pir:pirana_pir_operator",
        "//src/primihub/kernel/pir/operator/ypir:ypir_operator",
        "@com_github_glog_glog//:glog",
    ],
)
