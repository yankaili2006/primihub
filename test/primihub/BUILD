package(default_visibility = ["//visibility:public"])

# Common test deps
TEST_DEPS = [
    "@com_google_googletest//:gtest_main",
    "@com_github_glog_glog//:glog",
    "@com_google_absl//absl/strings",
]

# ================ Algorithm Tests ================

cc_test(
    name = "logistic_test",
    size = "small",
    srcs = ["algorithm/logistic_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:logistic_regression",
        "//src/primihub/data_store:data_store_factory",
        "//src/primihub/service/dataset:dataset_service",
        "//src/primihub/service/dataset/meta_service:localkv_impl",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "aby3_add_test",
    size = "small",
    srcs = ["algorithm/aby3_add_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:aby3_ml",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "aby3_mul_test",
    size = "small",
    srcs = ["algorithm/aby3_mul_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:aby3_ml",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "logistic_regression_test",
    size = "small",
    srcs = ["algorithm/logistic_regression_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:logistic_regression",
        "//src/primihub/data_store:data_store_factory",
        "//src/primihub/service/dataset:dataset_service",
        "//src/primihub/service/dataset/meta_service:localkv_impl",
        "//src/primihub/util/network:mem_channel",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "statistics_avg_test",
    size = "small",
    srcs = ["algorithm/statistics_avg_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:mpc_statistics",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "statistics_sum_test",
    size = "small",
    srcs = ["algorithm/statistics_sum_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:mpc_statistics",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "statistics_max_test",
    size = "small",
    srcs = ["algorithm/statistics_max_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:mpc_statistics",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "statistics_min_test",
    size = "small",
    srcs = ["algorithm/statistics_min_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:mpc_statistics",
    ],
    defines = ["GOOGLE_TEST"],
)

cc_test(
    name = "statistics_all_test",
    size = "small",
    srcs = ["algorithm/mpc_statistics_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/algorithm:mpc_statistics",
        "//src/primihub/service/dataset/meta_service:meta_service_factory",
    ],
    defines = ["GOOGLE_TEST"],
)

# ================ Common Tests ================

cc_test(
    name = "fixed_point_test",
    size = "small",
    srcs = ["common/type/fixed_point_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/common/type:fixed_point",
    ],
)

cc_test(
    name = "type_test",
    size = "small",
    srcs = ["common/type/type_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/common/type:type_def",
    ],
)

# ================ Utility Tests ================

cc_test(
    name = "aes_test",
    size = "small",
    srcs = ["util/crypto/aes_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/util/crypto:aes_util",
    ],
)

cc_test(
    name = "prng_test",
    size = "small",
    srcs = ["util/crypto/prng_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/util/crypto:prng_util",
    ],
)

cc_test(
    name = "channel_test",
    size = "small",
    srcs = ["util/network/socket/channel_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/util/network:network_channel",
    ],
)

# ================ PSI Tests ================

cc_test(
    name = "psi_operator_test",
    size = "small",
    srcs = ["kernel/psi_operator_test.cc"],
    deps = TEST_DEPS + [
        "//src/primihub/kernel/psi/operator:factory",
        "//src/primihub/kernel/psi/operator:base_psi_operator",
    ],
    defines = ["GOOGLE_TEST"],
)
