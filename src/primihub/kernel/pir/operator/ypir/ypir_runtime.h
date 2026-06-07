/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * YpirRuntime — thin C++ facade over the menonsamir/ypir C++ matrix
 * kernels (src/matmul.cpp, 7 SIMD-tuned matMulVecPacked variants
 * derived from simplepir's pir.c with packed-base layout).
 *
 * Purpose at this revision: ACTIVATION + SMOKE. Mirrors the
 * DoublePirRuntime / SimplePirRuntime pattern. Full YPIR algorithm
 * port (hint lifecycle + LWE query/answer + packing) is the multi-day
 * task 7.3 follow-up.
 *
 * Two compile modes selected by PIR_YPIR_RUNTIME_VENDORED, set by
 * the operator BUILD's select() when --define=enable_ypir_real=1:
 *
 *   * Defined: forwards facade calls into upstream matMulVecPacked.
 *     kYpirRuntimeVendored is true.
 *   * Undefined (default): every facade method returns retcode::FAIL
 *     with a clear "not vendored" message; kYpirRuntimeVendored is
 *     false.
 *
 * NOTE on the kernel call surface: upstream exposes 7 matMulVecPacked
 * variants (base + 2/4/6/8/8Alt/8Orig). They differ only in SIMD
 * unrolling; all produce identical output for identical input. The
 * smoke uses the unsuffixed `matMulVecPacked` (scalar) so the link
 * test does not depend on SIMD availability of the build host.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_RUNTIME_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_RUNTIME_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"

namespace primihub::pir::ypir {

extern const bool kYpirRuntimeVendored;

class YpirRuntime {
 public:
  static YpirRuntime& Instance();

  // Smoke test — calls upstream matMulVecPacked on an 8-row x 1-col
  // packed matrix with `a[i] = 0x01010101` (4 packed values of 1 per
  // entry) and `b = [1,1,1,1]`. Each row contributes 4 * (1 * 1) = 4
  // to its output bucket, so the expected `out` is [4,4,4,4,4,4,4,4].
  //
  // Returns SUCCESS only when the kernel link works and the math
  // matches. When kYpirRuntimeVendored is false, returns FAIL with a
  // populated `err`.
  retcode SmokeMatMulVecPacked(std::string* err);

 private:
  YpirRuntime() = default;
  YpirRuntime(const YpirRuntime&) = delete;
  YpirRuntime& operator=(const YpirRuntime&) = delete;
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_RUNTIME_H_
