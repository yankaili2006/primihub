/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/backend/avx2_backend.h"

namespace primihub::pir {

bool Avx2Backend::Available() const {
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
  // __builtin_cpu_supports is an x86-only builtin (absent on arm64 even under
  // gcc/clang). __builtin_cpu_init is implicitly called by it on first use;
  // safe to call from any thread after main has started.
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

}  // namespace primihub::pir
