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

# simple_pir_latency_bench — sibling of double_pir_latency_bench for
# SimplePirOperator. Driven by bench/simple_pir_persistence_bench.sh.
cc_binary(
    name = "simple_pir_latency_bench",
    srcs = ["simple_pir_latency_bench.cc"],
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "//src/primihub/kernel/pir/operator/simple_pir:simple_pir_operator",
        "@com_github_glog_glog//:glog",
    ],
)

# frodo_pir_latency_bench — sibling of double/simple_pir_latency_bench
# for FrodoPirOperator. Driven by bench/frodo_pir_latency_bench.sh.
# FrodoPIR is self-contained (no upstream override) so this builds in
# the default config — no --define / --override_repository flags needed.
cc_binary(
    name = "frodo_pir_latency_bench",
    srcs = ["frodo_pir_latency_bench.cc"],
    deps = [
        "//src/primihub/common:common_defination",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_api",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_database",
        "//src/primihub/kernel/pir/operator/frodo_pir:frodo_params",
        "@com_github_base64_cpp//:base64_lib",
    ],
)
