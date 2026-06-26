load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# SpiralPIR CUDA kernels (openspec change primihub-pir-cuda-tiptoe, task 2.1):
# GSW external product + Galois automorphism for the GPU backend.
#
# These compile only with a CUDA toolchain (nvcc + GPU). The default primihub
# build targets .50 (Xeon Broadwell, no GPU) and must not attempt to compile
# .cu sources, so the kernels are exposed here ONLY as a filegroup -- nothing in
# the default build depends on it, and no cc_library/cuda_library is declared
# yet. The bazel `cuda_library` target + `--define=disable_cuda=1` gate that
# wires these into the SpiralPIR operator's CUDA backend lands in task 2.3.
#
# Standalone build + validation (done on an RTX 5070 Ti / CUDA 13.2 when 2.1
# landed -- both kernels exact vs a CPU reference at poly_len=2048, crt=2):
#   nvcc -O2 -std=c++17 spiral_cuda_kernels.cu spiral_cuda_test.cu -I. \
#        -o spiral_cuda_test && ./spiral_cuda_test
filegroup(
    name = "spiral_cuda_srcs",
    srcs = [
        "spiral_cuda_kernels.cu",
        "spiral_cuda_kernels.h",
        "spiral_cuda_test.cu",
    ],
)

# Header-only interface, usable by CPU TUs that need the kernel declarations
# (e.g. the CUDA backend shim) without pulling in nvcc. The .cu implementation
# is linked separately by the task-2.3 cuda_library.
cc_library(
    name = "spiral_cuda_kernels_hdr",
    hdrs = ["spiral_cuda_kernels.h"],
)
