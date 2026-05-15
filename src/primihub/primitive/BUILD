package(default_visibility = ["//visibility:public"])

cc_library(
    name = "primitive_lib",
    srcs = glob(["**/*.cc"]),
    hdrs = glob(["**/*.h"]),
    deps = [
        "@openssl//:openssl",
        "@com_github_glog_glog//:glog",
    ],
    copts = ["-I/usr/include", "-L/usr/lib"],
)

cc_library(
    name = "oprf",
    srcs = ["oprf/oprf.cc"],
    hdrs = ["oprf/oprf.h"],
    deps = [
        "@openssl//:openssl",
    ],
)
