/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * YpirRuntime implementation.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_runtime.h"

#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>

#include <glog/logging.h>

#ifdef PIR_YPIR_RUNTIME_VENDORED
extern "C" {
// Mirrors upstream src/matmul.cpp `extern "C"` declarations. Upstream
// has no public header; we forward-declare the scalar variant here
// (other 6 SIMD variants share this signature). Keeping the forward
// inline avoids #include-time exposure of upstream globals.
void matMulVecPacked(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     size_t aRows, size_t aCols);
}  // extern "C"
#endif  // PIR_YPIR_RUNTIME_VENDORED

namespace primihub::pir::ypir {

#ifdef PIR_YPIR_RUNTIME_VENDORED
const bool kYpirRuntimeVendored = true;
#else
const bool kYpirRuntimeVendored = false;
#endif

namespace {
std::mutex& RuntimeMutex() {
  static std::mutex m;
  return m;
}
}  // namespace

YpirRuntime& YpirRuntime::Instance() {
  static YpirRuntime instance;
  return instance;
}

retcode YpirRuntime::SmokeMatMulVecPacked(std::string* err) {
  std::lock_guard<std::mutex> lk(RuntimeMutex());

#ifndef PIR_YPIR_RUNTIME_VENDORED
  if (err) {
    *err =
        "YpirRuntime: not vendored. Build with "
        "--define=enable_ypir_real=1 and provide the @ypir bazel "
        "override pointing at menonsamir/ypir (see "
        "openspec/changes/primihub-pir-multi-algo Phase 6 task 7.3).";
  }
  return retcode::FAIL;
#else
  // 8 rows x 1 column packed matrix. Each `a[i]` packs 4 8-bit values
  // (COMPRESSION=4, BASIS=8 in upstream matmul.cpp). With
  // `a[i] = 0x01010101` each row contributes 4 * 1 = 4 to out[i] when
  // `b = [1,1,1,1]` (b has aCols * COMPRESSION = 4 entries).
  static constexpr size_t kRows = 8;
  static constexpr size_t kCols = 1;
  uint32_t a[kRows * kCols];
  for (size_t i = 0; i < kRows * kCols; ++i) {
    a[i] = 0x01010101u;
  }
  uint32_t b[4] = {1, 1, 1, 1};
  uint32_t out[kRows] = {0, 0, 0, 0, 0, 0, 0, 0};

  matMulVecPacked(out, a, b, kRows, kCols);

  for (size_t i = 0; i < kRows; ++i) {
    if (out[i] != 4u) {
      if (err) {
        std::ostringstream oss;
        oss << "YpirRuntime::SmokeMatMulVecPacked: kernel link works "
            << "but math diverges at out[" << i << "]: got " << out[i]
            << " want 4. Check WORKSPACE_GITHUB @ypir pin a73e550a "
            << "and src/matmul.cpp matMulVecPacked semantics (COMPRESSION/"
            << "BASIS may have shifted).";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
  }
  LOG(INFO) << "YpirRuntime::SmokeMatMulVecPacked: matMulVecPacked "
            << "kernel link validated (8x1 packed-1 case)";
  return retcode::SUCCESS;
#endif  // PIR_YPIR_RUNTIME_VENDORED
}

}  // namespace primihub::pir::ypir
