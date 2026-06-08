load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# double_pir_latency_bench — minimal driver for DoublePirOperator that
# emits per-N latency in CSV or human-readable form. Driven by
# bench/double_pir_latency_bench.sh which sweeps a range of N values.
#
# Build with vendored mode (the only mode where DoublePIR runs):
#   bazel build --config=linux_x86_64 \
#               --define=enable_pir_core_real=1 \
#               --define=enable_double_pir_real=1 \
#               --override_repository=simplepir=<path> \
#               //src/primihub/kernel/pir/bench:double_pir_latency_bench
cc_binary(
    name = "double_pir_latency_bench",
    srcs = ["double_pir_latency_bench.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/double_pir:double_pir_operator",
        "@com_github_glog_glog//:glog",
    ],
)
