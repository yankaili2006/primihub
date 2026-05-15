cc_test(
    name = "bench_mpc",
    srcs = ["bench_mpc.cc"],
    deps = [
        "//src/primihub/protocol:protocol_aby3_lib",
        "//src/primihub/operator:aby3_operator",
        "//src/primihub/util/network:network_lib",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
