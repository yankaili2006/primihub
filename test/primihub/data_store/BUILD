package(default_visibility = ["//visibility:public"])

cc_test(
    name = "csv_driver_test",
    size = "small",
    srcs = ["csv_driver_test.cc"],
    deps = [
        "//src/primihub/data_store:data_store_lib",
        "//src/primihub/data_store/csv:csv_driver",
        "//src/primihub/data_store:base_driver",
        "@com_google_googletest//:gtest_main",
        "@com_github_glog_glog//:glog",
        "@arrow",
    ],
    linkopts = ["-lpthread"],
)

cc_test(
    name = "parquet_driver_test",
    size = "small",
    srcs = ["parquet_driver_test.cc"],
    deps = [
        "//src/primihub/data_store:data_store_lib",
        "//src/primihub/data_store/parquet:parquet_driver",
        "//src/primihub/data_store:base_driver",
        "@com_google_googletest//:gtest_main",
        "@com_github_glog_glog//:glog",
        "@arrow",
    ],
    linkopts = ["-lpthread"],
)

cc_test(
    name = "sqlite_driver_test",
    size = "small",
    srcs = ["sqlite_driver_test.cc"],
    deps = [
        "//src/primihub/data_store:data_store_lib",
        "//src/primihub/data_store/sqlite:sqlite_driver",
        "//src/primihub/data_store:base_driver",
        "@com_google_googletest//:gtest_main",
        "@com_github_glog_glog//:glog",
        "@arrow",
    ],
    linkopts = ["-lpthread"],
)
