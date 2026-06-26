load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

# SpiralPIR CUDA kernels (openspec change primihub-pir-cuda-tiptoe, tasks
# 2.1/2.3): GSW external product + Galois automorphism for the GPU backend.
#
# The .cu compiles only with a CUDA toolchain (nvcc + GPU). .50 (Broadwell, no
# GPU) must not compile it, so:
#   - spiral_cuda_kernels_hdr: header-only interface, builds everywhere.
#   - spiral_cuda_kernels: the real nvcc-compiled library, linked ONLY under
#     --define=enable_cuda=1 (GPU hosts). Under the default config its srcs are
#     empty (header-only), so the genrule/nvcc never runs on .50 or in CI.
#
# Standalone build + validation (RTX 5070 Ti / CUDA 13.2 -- both kernels exact
# vs a CPU reference at poly_len=2048, crt=2):
#   nvcc -O2 -std=c++17 spiral_cuda_kernels.cu spiral_cuda_test.cu -I. \
#        -o spiral_cuda_test && ./spiral_cuda_test
# NOTE: the bazel enable_cuda path below is the integration structure; it
# requires nvcc in PATH on a GPU host and is NOT exercised in CI.
filegroup(
    name = "spiral_cuda_srcs",
    srcs = [
        "spiral_cuda_kernels.cu",
        "spiral_cuda_kernels.h",
        "spiral_cuda_test.cu",
    ],
)

cc_library(
    name = "spiral_cuda_kernels_hdr",
    hdrs = ["spiral_cuda_kernels.h"],
)

# nvcc-compiled object (GPU hosts only; opt-in). manual + no-sandbox so it is
# never built by //... and only fires when :spiral_cuda_kernels is built under
# --define=enable_cuda=1.
genrule(
    name = "spiral_cuda_kernels_obj",
    srcs = [
        "spiral_cuda_kernels.cu",
        "spiral_cuda_kernels.h",
    ],
    outs = ["spiral_cuda_kernels.pic.o"],
    cmd = "nvcc -O2 -std=c++17 --compiler-options -fPIC -c " +
          "$(location spiral_cuda_kernels.cu) " +
          "-I$$(dirname $(location spiral_cuda_kernels.h)) -o $@",
    tags = [
        "manual",
        "no-sandbox",
    ],
)

cc_library(
    name = "spiral_cuda_kernels",
    srcs = select({
        "//src/primihub/kernel/pir/operator/backend:enable_cuda": [
            ":spiral_cuda_kernels_obj",
        ],
        "//conditions:default": [],
    }),
    hdrs = ["spiral_cuda_kernels.h"],
    linkopts = select({
        "//src/primihub/kernel/pir/operator/backend:enable_cuda": [
            "-L/usr/local/cuda/lib64",
            "-lcudart",
        ],
        "//conditions:default": [],
    }),
    tags = ["manual"],
)
