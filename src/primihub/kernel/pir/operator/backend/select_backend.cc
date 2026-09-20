/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/backend/backend.h"

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/backend/avx2_backend.h"
#include "src/primihub/kernel/pir/operator/backend/cpu_backend.h"
#ifndef DISABLE_CUDA
#include "src/primihub/kernel/pir/operator/backend/cuda_backend.h"
#endif

namespace primihub::pir {
namespace {

// Try the candidate backend if it's in the algorithm-supported set and the
// host actually has it. Logs a one-line decision trace.
std::unique_ptr<PirBackend> TryBackend(
    Backend type,
    const std::set<Backend>& supported) {
  if (!supported.count(type)) return nullptr;
  std::unique_ptr<PirBackend> b;
  switch (type) {
    case Backend::CPU:
      b = std::make_unique<CpuBackend>();
      break;
    case Backend::AVX2:
      b = std::make_unique<Avx2Backend>();
      break;
    case Backend::CUDA:
#ifndef DISABLE_CUDA
      b = std::make_unique<CudaBackend>();
#else
      return nullptr;
#endif
      break;
    default:
      return nullptr;
  }
  if (b && b->Available()) {
    return b;
  }
  return nullptr;
}

}  // namespace

std::unique_ptr<PirBackend> SelectBackend(
    Backend preferred,
    const std::set<Backend>& supported) {
  // Auto: prefer fastest available
  if (preferred == Backend::AUTO) {
    for (Backend cand : {Backend::CUDA, Backend::AVX2, Backend::CPU}) {
      if (auto b = TryBackend(cand, supported)) {
        VLOG(2) << "SelectBackend(AUTO) → " << b->Name();
        return b;
      }
    }
    LOG(ERROR) << "SelectBackend(AUTO): no backend available";
    return nullptr;
  }
  // Explicit preference first
  if (auto b = TryBackend(preferred, supported)) {
    VLOG(2) << "SelectBackend(" << ToString(preferred)
            << ") satisfied by " << b->Name();
    return b;
  }
  // Fallback: walk preference order (faster first) starting after preferred
  LOG(INFO) << "SelectBackend: preferred " << ToString(preferred)
            << " unavailable, falling back";
  for (Backend cand : {Backend::CUDA, Backend::AVX2, Backend::CPU}) {
    if (cand == preferred) continue;
    if (auto b = TryBackend(cand, supported)) {
      VLOG(2) << "SelectBackend fallback → " << b->Name();
      return b;
    }
  }
  LOG(ERROR) << "SelectBackend: no fallback backend available";
  return nullptr;
}

}  // namespace primihub::pir
