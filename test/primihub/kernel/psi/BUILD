package(default_visibility = ["//visibility:public"])

cc_test(
    name = "psi_util_test",
    size = "small",
    srcs = ["psi_util_test.cc"],
    deps = [
        "//src/primihub/kernel/psi:psi_util",
        "@com_google_googletest//:gtest_main",
        "@arrow",
    ],
)

cc_test(
    name = "ecdh_psi_mock_test",
    size = "small",
    srcs = ["ecdh_psi_mock_test.cc"],
    deps = [
        "//test/primihub/util/network:mock_channel",
        "//src/primihub/kernel/psi/operator:ecdh_psi_operator",
        "//src/primihub/kernel/psi/operator:factory",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "base_psi_test",
    size = "small",
    srcs = ["base_psi_test.cc"],
    deps = [
        "//test/primihub/util/network:mock_channel",
        "//src/primihub/kernel/psi/operator:factory",
        "//src/primihub/kernel/psi/operator:base_psi_operator",
        "@com_google_googletest//:gtest_main",
    ],
)
