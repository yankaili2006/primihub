package(default_visibility = ["//visibility:public"])

# CUDA build gates (openspec change primihub-pir-cuda-tiptoe, task 2.3).
#
# disable_cuda (--define=disable_cuda=1): strips the CUDA backend from :backend
# and sets the DISABLE_CUDA define so select_backend.cc compiles out its CUDA
# codepath (#ifndef DISABLE_CUDA). Use on CPU-only hosts that don't even want
# the CudaBackend stub linked.
#
# enable_cuda (--define=enable_cuda=1): turns on the nvcc-built kernel
# cuda_library targets (//...spiral_pir/cuda, //...double_pir/cuda). GPU hosts
# only (needs nvcc); off by default so .50 / CI never compile .cu.
config_setting(
    name = "disable_cuda",
    define_values = {"disable_cuda": "1"},
)

config_setting(
    name = "enable_cuda",
    define_values = {"enable_cuda": "1"},
)

cc_library(
    name = "backend",
    hdrs = [
        "backend.h",
        "cpu_backend.h",
        "avx2_backend.h",
    ] + select({
        ":disable_cuda": [],
        "//conditions:default": ["cuda_backend.h"],
    }),
    srcs = [
        "avx2_backend.cc",
        "select_backend.cc",
    ] + select({
        ":disable_cuda": [],
        "//conditions:default": ["cuda_backend.cc"],
    }),
    defines = select({
        # When disabled, set DISABLE_CUDA so select_backend.cc drops the CUDA
        # branch and the cuda_backend.{h,cc} stub is not compiled in.
        ":disable_cuda": ["DISABLE_CUDA"],
        "//conditions:default": [],
    }),
    deps = [
        "//src/primihub/kernel/pir:common_def",
        "@com_github_glog_glog//:glog",
    ],
)
