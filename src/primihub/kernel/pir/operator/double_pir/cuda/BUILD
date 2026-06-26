load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# DoublePIR CUDA kernel (openspec change primihub-pir-cuda-tiptoe, task 2.2):
# the LWE matrix multiply C = A*B mod 2^32 (the per-query Answer hot path).
#
# Compiles only with a CUDA toolchain (nvcc + GPU). The default primihub build
# targets .50 (no GPU) and must not compile .cu sources, so the kernel is
# exposed here ONLY as a filegroup + a header-only interface lib; nothing in the
# default build depends on the .cu. The bazel `cuda_library` target +
# `--define=disable_cuda=1` gate that wires this into the DoublePIR CUDA backend
# lands in task 2.3.
#
# Standalone build + validation (RTX 5070 Ti / CUDA 13.2 when 2.2 landed -- exact
# vs a CPU reference for batched, matrix-vector, and ragged dims):
#   nvcc -O2 -std=c++17 double_cuda_kernels.cu double_cuda_test.cu -I. \
#        -o double_cuda_test && ./double_cuda_test
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
