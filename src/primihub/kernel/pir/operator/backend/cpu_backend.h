/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CPU_BACKEND_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CPU_BACKEND_H_

#include "src/primihub/kernel/pir/operator/backend/backend.h"

namespace primihub::pir {

class CpuBackend : public PirBackend {
 public:
  std::string Name() const override { return "cpu"; }
  Backend Type() const override { return Backend::CPU; }
  bool Available() const override { return true; }
  PerfHints GetPerfHints() const override {
    PerfHints h;
    h.expected_latency_us_per_query_at_1e7 = 200000;  // 200ms baseline
    h.expected_throughput_qps_at_1e7 = 5;
    h.has_simd = false;
    h.note = "Scalar baseline";
    return h;
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CPU_BACKEND_H_
