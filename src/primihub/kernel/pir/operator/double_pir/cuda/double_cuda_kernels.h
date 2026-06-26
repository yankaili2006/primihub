/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA kernel for the DoublePIR (USENIX'23) LWE matrix multiply (openspec
 * change primihub-pir-cuda-tiptoe, task 2.2): the per-query Answer hot path,
 * C = A * B over the LWE modulus q = 2^32 (uint32 arithmetic wraps naturally).
 * This is the GPU analogue of the AVX2 MulVecPacked path; it covers both the
 * matrix-vector Answer (cols = 1) and small batched queries (cols > 1).
 *
 * Built only with a CUDA toolchain (nvcc + GPU); excluded from the default
 * CPU-only build (.50 has no GPU) -- see the cuda/ BUILD. The packed/squished
 * DB layout and Montgomery reduction (for prime moduli) are perf follow-ups;
 * the production path reuses pir-acc/SIGMA's CUDA primitives.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_CUDA_DOUBLE_CUDA_KERNELS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_CUDA_DOUBLE_CUDA_KERNELS_H_

#include <cstddef>
#include <cstdint>

namespace primihub::pir::doublepir::cuda {

// Whether a usable CUDA device is present.
bool CudaAvailable();

// C = A * B  mod 2^32, where A is rows x inner, B is inner x cols, C is
// rows x cols, all row-major uint32. Tiled shared-memory matmul; accumulation
// is in uint32 so it wraps mod 2^32 (the DoublePIR LWE modulus). Host pointers.
void LweMatMulMod2Pow32(std::uint32_t* c, const std::uint32_t* a,
                        const std::uint32_t* b, std::size_t rows,
                        std::size_t inner, std::size_t cols);

}  // namespace primihub::pir::doublepir::cuda

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_CUDA_DOUBLE_CUDA_KERNELS_H_
