/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirRuntime implementation.
 */
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_runtime.h"

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

#include <glog/logging.h>

#ifdef PIR_SIMPLE_PIR_RUNTIME_VENDORED
extern "C" {
typedef uint32_t Elem;
void matMul(Elem* out, const Elem* a, const Elem* b,
            size_t aRows, size_t aCols, size_t bCols);
}  // extern "C"
#endif  // PIR_SIMPLE_PIR_RUNTIME_VENDORED

namespace primihub::pir::simple_pir {

#ifdef PIR_SIMPLE_PIR_RUNTIME_VENDORED
const bool kSimplePirRuntimeVendored = true;
#else
const bool kSimplePirRuntimeVendored = false;
#endif

namespace {
std::mutex& RuntimeMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

SimplePirRuntime& SimplePirRuntime::Instance() {
  static SimplePirRuntime instance;
  return instance;
}

retcode SimplePirRuntime::SmokeMatMul(std::string* err) {
  std::lock_guard<std::mutex> lk(RuntimeMutex());

#ifndef PIR_SIMPLE_PIR_RUNTIME_VENDORED
  if (err) {
    *err =
        "SimplePirRuntime: not vendored. Build with "
        "--define=enable_simple_pir_real=1 and provide the @simplepir "
        "bazel override pointing at ahenzinger/simplepir (see "
        "openspec/changes/primihub-pir-multi-algo Phase 6 task 7.2).";
  }
  return retcode::FAIL;
#else
  // 3x3 * 3x3 matMul where a[r][c] = r * 3 + c, b = identity.
  // Expected: out == a.
  static constexpr size_t kN = 3;
  uint32_t a[kN * kN];
  for (size_t r = 0; r < kN; ++r) {
    for (size_t c = 0; c < kN; ++c) {
      a[r * kN + c] = static_cast<uint32_t>(r * kN + c);
    }
  }
  uint32_t b[kN * kN] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  uint32_t out[kN * kN] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

  matMul(out, a, b, kN, kN, kN);

  for (size_t r = 0; r < kN; ++r) {
    for (size_t c = 0; c < kN; ++c) {
      const uint32_t want = a[r * kN + c];
      if (out[r * kN + c] != want) {
        if (err) {
          std::ostringstream oss;
          oss << "SimplePirRuntime::SmokeMatMul: kernel link works but "
              << "math diverges at out[" << r << "][" << c << "]: got "
              << out[r * kN + c] << " want " << want
              << ". Check WORKSPACE_GITHUB @simplepir pin e9020b03 and "
              << "pir/pir.c matMul semantics.";
          *err = oss.str();
        }
        return retcode::FAIL;
      }
    }
  }
  LOG(INFO) << "SimplePirRuntime::SmokeMatMul: matMul kernel link "
            << "validated (3x3 identity case)";
  return retcode::SUCCESS;
#endif  // PIR_SIMPLE_PIR_RUNTIME_VENDORED
}

}  // namespace primihub::pir::simple_pir
