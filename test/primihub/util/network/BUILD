package(default_visibility = ["//visibility:public"])

cc_test(
    name = "mock_channel_test",
    size = "small",
    srcs = ["mock_channel_test.cc"],
    deps = [
        ":mock_channel",
        "//src/primihub/kernel/psi/operator:factory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mock_channel",
    hdrs = ["mock_channel.h"],
    deps = [
        "//src/primihub/util/network:communication_lib",
        "@com_google_googletest//:gtest",
    ],
)
