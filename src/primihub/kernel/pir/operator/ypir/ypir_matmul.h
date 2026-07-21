/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_matmul — C++ port of upstream menonsamir/ypir@a73e550a
 * src/matmul.rs. Thin shape-validating dispatcher over the four
 * `matMulVecPacked{,2,4,8}` kernels from @ypir//:ypir_matmul_kernels.
 *
 * Position in the port plan (docs/pir/ypir-port-plan.md):
 *   chunk 2 — after `transpose.rs` (chunk 1, commit d1ffde07).
 *   No dependency on the Spiral C++ port (which is blocking
 *   chunks 3, 7, 9-12). Only depends on the matmul kernels
 *   already linked by the @ypir bazel override.
 *
 * Vendored gating mirrors ypir_runtime.cc:
 *   * --define=enable_ypir_real=1 + @ypir override: real path
 *     calls into matMulVecPacked* and computes the product.
 *   * default: returns retcode::FAIL with a populated `err`
 *     guiding the caller to the activation flag.
 *
 * The packed-base layout is upstream's: each `a` entry packs
 * BASIS=8-bit values via COMPRESSION=4, so the matmul caller
 * must keep `a_cols * 4 == b_rows`. `b_cols` must be in
 * {1, 2, 4, 8} — these correspond to the four kernel variants;
 * b_cols=6 was disabled upstream (commented out in matmul.rs)
 * and we preserve that.
 *
 * Output buffer convention: upstream requires
 *   `out.len() >= (a_rows + 8) * b_cols`
 * for SIMD tail-write slack. We enforce that explicitly.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MATMUL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MATMUL_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"

namespace primihub::pir::ypir {

// True when built with --define=enable_ypir_real=1 (linked against
// @ypir//:ypir_matmul_kernels). Mirrors kYpirRuntimeVendored.
extern const bool kYpirMatmulVendored;

// Dispatch to the appropriate `matMulVecPacked{,2,4,8}` kernel from
// @ypir//:ypir_matmul_kernels. On precondition failure or non-
// vendored builds, returns retcode::FAIL and populates `err` (if
// non-null) with a diagnostic. On success, `out` holds the product.
//
// Preconditions (FAIL with diagnostic if violated):
//   * out != nullptr, a != nullptr, b != nullptr
//   * a_len == a_rows * a_cols
//   * b_len == b_rows * b_cols
//   * a_cols * 4 == b_rows (packed-base shape contract)
//   * b_cols ∈ {1, 2, 4, 8}
//   * out_len >= (a_rows + 8) * b_cols (SIMD tail slack)
retcode MatMulVecPacked(uint32_t* out, std::size_t out_len,
                        const uint32_t* a, std::size_t a_len,
                        const uint32_t* b, std::size_t b_len,
                        std::size_t a_rows, std::size_t a_cols,
                        std::size_t b_rows, std::size_t b_cols,
                        std::string* err);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MATMUL_H_
