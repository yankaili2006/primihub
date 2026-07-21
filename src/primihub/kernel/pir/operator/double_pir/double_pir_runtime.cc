/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * DoublePirRuntime implementation. See header for the two-mode contract.
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir_runtime.h"

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

#include <glog/logging.h>

#ifdef PIR_DOUBLE_PIR_RUNTIME_VENDORED
// Upstream simplepir exposes its kernels as plain C functions in
// pir/pir.h. We forward-declare them here rather than #including the
// upstream header so that this TU compiles either way: when the include
// path resolves the upstream BUILD wrapper adds the cc_library to
// link-time deps, and the symbols resolve. Keeping the forwards inline
// (instead of #include) also avoids dragging stdio.h into the rest of
// the primihub PIR subtree.
extern "C" {
typedef uint32_t Elem;
void matMulVec(Elem* out, const Elem* a, const Elem* b,
               size_t aRows, size_t aCols);
}  // extern "C"
#endif  // PIR_DOUBLE_PIR_RUNTIME_VENDORED

namespace primihub::pir::double_pir {

#ifdef PIR_DOUBLE_PIR_RUNTIME_VENDORED
const bool kDoublePirRuntimeVendored = true;
#else
const bool kDoublePirRuntimeVendored = false;
#endif

namespace {
std::mutex& RuntimeMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

DoublePirRuntime& DoublePirRuntime::Instance() {
  static DoublePirRuntime instance;
  return instance;
}

retcode DoublePirRuntime::SmokeMatMulVec(std::string* err) {
  std::lock_guard<std::mutex> lk(RuntimeMutex());

#ifndef PIR_DOUBLE_PIR_RUNTIME_VENDORED
  if (err) {
    *err =
        "DoublePirRuntime: not vendored. Build with "
        "--define=enable_double_pir_real=1 and provide the @simplepir "
        "bazel override pointing at ahenzinger/simplepir (see "
        "openspec/changes/primihub-pir-multi-algo design.md Phase 4).";
  }
  return retcode::FAIL;
#else
  // 4x4 matrix `a` with row r = [r, r+1, r+2, r+3].
  // Vector `b` = [1, 1, 1, 1].
  // Expected `out[r]` = sum of row = 4r + 6.
  static constexpr size_t kRows = 4;
  static constexpr size_t kCols = 4;
  uint32_t a[kRows * kCols];
  for (size_t r = 0; r < kRows; ++r) {
    for (size_t c = 0; c < kCols; ++c) {
      a[r * kCols + c] = static_cast<uint32_t>(r + c);
    }
  }
  uint32_t b[kCols] = {1, 1, 1, 1};
  uint32_t out[kRows] = {0, 0, 0, 0};

  matMulVec(out, a, b, kRows, kCols);

  for (size_t r = 0; r < kRows; ++r) {
    const uint32_t want = static_cast<uint32_t>(4 * r + 6);
    if (out[r] != want) {
      if (err) {
        std::ostringstream oss;
        oss << "DoublePirRuntime::SmokeMatMulVec: kernel link works but "
            << "math diverges at row " << r << ": got " << out[r]
            << " want " << want
            << ". The @simplepir upstream may have changed its uint32 "
            << "matMulVec semantics; check WORKSPACE_GITHUB pin "
            << "e9020b03 and pir/pir.c.";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
  }
  LOG(INFO) << "DoublePirRuntime::SmokeMatMulVec: matMul kernel link "
            << "validated (4x4 * 4 case)";
  return retcode::SUCCESS;
#endif  // PIR_DOUBLE_PIR_RUNTIME_VENDORED
}

}  // namespace primihub::pir::double_pir
