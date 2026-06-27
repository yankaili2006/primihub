load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# DoublePIR CUDA kernel (openspec change primihub-pir-cuda-tiptoe, tasks
# 2.2/2.3): the LWE matrix multiply C = A*B mod 2^32 (the per-query Answer hot
# path).
#
# The .cu compiles only with a CUDA toolchain (nvcc + GPU). .50 (no GPU) must
# not compile it, so:
#   - double_cuda_kernels_hdr: header-only interface, builds everywhere.
#   - double_cuda_kernels: the real nvcc-compiled library, linked ONLY under
#     --define=enable_cuda=1 (GPU hosts). Default config: srcs empty.
#
# Standalone build + validation (RTX 5070 Ti / CUDA 13.2 -- exact vs a CPU
# reference for batched, matrix-vector, and ragged dims):
#   nvcc -O2 -std=c++17 double_cuda_kernels.cu double_cuda_test.cu -I. \
#        -o double_cuda_test && ./double_cuda_test
# NOTE: the bazel enable_cuda path requires nvcc in PATH on a GPU host and is
# NOT exercised in CI.
filegroup(
    name = "double_cuda_srcs",
    srcs = [
        "double_cuda_kernels.cu",
        "double_cuda_kernels.h",
        "double_cuda_test.cu",
    ],
)

cc_library(
    name = "double_cuda_kernels_hdr",
    hdrs = ["double_cuda_kernels.h"],
)

genrule(
    name = "double_cuda_kernels_obj",
    srcs = [
        "double_cuda_kernels.cu",
        "double_cuda_kernels.h",
    ],
    outs = ["double_cuda_kernels.pic.o"],
    cmd = "nvcc -O2 -std=c++17 --compiler-options -fPIC -c " +
          "$(location double_cuda_kernels.cu) " +
          "-I$$(dirname $(location double_cuda_kernels.h)) -o $@",
    tags = [
        "manual",
        "no-sandbox",
    ],
)

cc_library(
    name = "double_cuda_kernels",
    srcs = select({
        "//src/primihub/kernel/pir/operator/backend:enable_cuda": [
            ":double_cuda_kernels_obj",
        ],
        "//conditions:default": [],
    }),
    hdrs = ["double_cuda_kernels.h"],
    linkopts = select({
        "//src/primihub/kernel/pir/operator/backend:enable_cuda": [
            "-L/usr/local/cuda/lib64",
            "-lcudart",
        ],
        "//conditions:default": [],
    }),
    tags = ["manual"],
)
