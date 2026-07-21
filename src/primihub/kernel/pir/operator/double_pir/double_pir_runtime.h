/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * DoublePirRuntime — thin C++ facade over the ahenzinger/simplepir C
 * matrix-multiplication kernels (the upstream's pir/pir.c matMul,
 * matMulVec, matMulVecPacked, transpose).
 *
 * Purpose at this revision: ACTIVATION + SMOKE. Validates that the
 * @simplepir//:simplepir_c_kernels cc_library wired by
 * thirdparty/pir/BUILD.simplepir links cleanly into the primihub PIR
 * subtree under --define=enable_double_pir_real=1, and that the C
 * matMul kernel returns the correct uint32 product on a small known
 * matrix (so we have a tight signal when the kernel link breaks or the
 * upstream API changes). This is the foothold for the full DoublePIR
 * algorithm port (openspec/changes/primihub-pir-multi-algo task 5.5),
 * which is multi-day work and is NOT in scope of this commit.
 *
 * Two compile modes selected by PIR_DOUBLE_PIR_RUNTIME_VENDORED, set by
 * the operator BUILD's `select()` when --define=enable_double_pir_real=1:
 *
 *   * Defined: includes the upstream pir.h forward decls and calls
 *     matMul on a 4x4 / 4x1 / 4 matrix triplet to confirm the kernel
 *     executes. kDoublePirRuntimeVendored is true.
 *
 *   * Undefined (default): every facade method returns retcode::FAIL
 *     with a clear "not vendored" message. The operator is still
 *     constructible and skeleton tests still pass; this matches the
 *     SpiralRuntime stub-vs-real pattern (see spiral_runtime.h).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_RUNTIME_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_RUNTIME_H_

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"

namespace primihub::pir::double_pir {

// True iff this build links against @simplepir//:simplepir_c_kernels.
// When false, every facade method returns retcode::FAIL.
extern const bool kDoublePirRuntimeVendored;

// Process-wide singleton wrapping all calls into the simplepir C
// kernels. The kernels themselves are pure functions (no global state),
// but we serialize all facade calls with a single std::mutex so future
// stateful additions (LWE secret material, hint cache) do not race.
class DoublePirRuntime {
 public:
  static DoublePirRuntime& Instance();

  // Smoke test — fills a 4-row x 4-col uint32 matrix `a` and a 4-element
  // uint32 vector `b` with a known pattern, calls upstream matMulVec to
  // produce a 4-element output, and verifies the result against the
  // expected dot products computed in-line. Returns retcode::SUCCESS
  // when the kernel link works and the math matches. When
  // kDoublePirRuntimeVendored is false, returns retcode::FAIL with a
  // populated `err`.
  //
  // This is intentionally narrow: it validates the integration boundary
  // (cc_library link + extern "C" symbol resolution + uint32 arithmetic)
  // without claiming any DoublePIR algorithm semantics. The full
  // protocol implementation lands as a follow-up to task 5.5.
  retcode SmokeMatMulVec(std::string* err);

 private:
  DoublePirRuntime() = default;
  DoublePirRuntime(const DoublePirRuntime&) = delete;
  DoublePirRuntime& operator=(const DoublePirRuntime&) = delete;
};

}  // namespace primihub::pir::double_pir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_RUNTIME_H_
