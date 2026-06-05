package(default_visibility = ["//visibility:public"])

cc_library(
    name = "backend",
    hdrs = [
        "backend.h",
        "cpu_backend.h",
        "avx2_backend.h",
    ] + select({
        "//conditions:default": ["cuda_backend.h"],
    }),
    srcs = [
        "avx2_backend.cc",
        "select_backend.cc",
    ] + select({
        "//conditions:default": ["cuda_backend.cc"],
    }),
    defines = select({
        # Disable CUDA codepaths when building without HAVE_CUDA. We compile
        # the cuda_backend.{h,cc} unconditionally for symbol presence; the
        # impl returns Available()=false when HAVE_CUDA is not defined.
        "//conditions:default": [],
    }),
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "@com_github_glog_glog//:glog",
    ],
)
