/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirRuntime — thin C++ facade over the ahenzinger/simplepir C
 * matrix-multiplication kernels (pir/pir.c).
 *
 * Purpose at this revision: ACTIVATION + SMOKE. Mirrors the
 * DoublePirRuntime pattern (commit dc037df7) so SimplePirOperator
 * also has a definitive runtime signal when the @simplepir cc_library
 * link breaks. The full SimplePIR algorithm port (LWE secret + hint +
 * answer + decode) is the multi-day task 7.2 follow-up.
 *
 * Two compile modes selected by PIR_SIMPLE_PIR_RUNTIME_VENDORED, set by
 * the operator BUILD's select() when --define=enable_simple_pir_real=1:
 *
 *   * Defined: forwards facade calls into the real upstream C kernel.
 *     kSimplePirRuntimeVendored is true.
 *   * Undefined (default): every facade method returns retcode::FAIL
 *     with a clear "not vendored" message; kSimplePirRuntimeVendored
 *     is false.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_RUNTIME_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_RUNTIME_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"

namespace primihub::pir::simple_pir {

extern const bool kSimplePirRuntimeVendored;

class SimplePirRuntime {
 public:
  static SimplePirRuntime& Instance();

  // Smoke test — calls upstream matMul on a 3-row x 3-col uint32 matrix
  // multiplied with itself (essentially A^2) and verifies the 9 output
  // values against the in-line computed expected. Returns SUCCESS only
  // when the kernel link works and the math matches. When
  // kSimplePirRuntimeVendored is false, returns FAIL with a populated
  // `err`.
  retcode SmokeMatMul(std::string* err);

 private:
  SimplePirRuntime() = default;
  SimplePirRuntime(const SimplePirRuntime&) = delete;
  SimplePirRuntime& operator=(const SimplePirRuntime&) = delete;
};

}  // namespace primihub::pir::simple_pir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_RUNTIME_H_
