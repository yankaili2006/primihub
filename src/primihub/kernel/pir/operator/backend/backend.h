/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_BACKEND_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_BACKEND_H_

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include "src/primihub/kernel/pir/common.h"

namespace primihub::pir {

// Performance hints per backend, used by the Selector to rank algorithm
// candidates and by the operator at construction time to decide between
// available backends. Numbers are baseline estimates measured at 1e7-row
// scale on 256-bit elements; algorithms MAY override with their own values.
struct PerfHints {
  uint64_t expected_latency_us_per_query_at_1e7 = 0;
  uint64_t expected_throughput_qps_at_1e7 = 0;
  bool has_simd = false;
  std::string note;
};

class PirBackend {
 public:
  virtual ~PirBackend() = default;
  virtual std::string Name() const = 0;
  virtual Backend Type() const = 0;
  // Probed at runtime — CPU always true; AVX2 via __builtin_cpu_supports;
  // CUDA via cudaGetDeviceCount when compiled with HAVE_CUDA, else false.
  virtual bool Available() const = 0;
  virtual PerfHints GetPerfHints() const = 0;
};

// Picks the best backend available on this host that is also in the
// algorithm-supported set. Semantics:
//   - If preferred is AUTO: prefer CUDA > AVX2 > CPU among (available ∩ supported)
//   - If preferred is concrete and (available ∧ supported): use it
//   - Otherwise: fall back to next-best per AUTO order
// Returns nullptr only when none of the supported backends are available
// (which should be impossible for any algorithm that declares CPU support).
std::unique_ptr<PirBackend> SelectBackend(
    Backend preferred,
    const std::set<Backend>& algorithm_supported);

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_BACKEND_H_
