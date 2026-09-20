/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_AVX2_BACKEND_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_AVX2_BACKEND_H_

#include "src/primihub/kernel/pir/operator/backend/backend.h"

namespace primihub::pir {

class Avx2Backend : public PirBackend {
 public:
  std::string Name() const override { return "avx2"; }
  Backend Type() const override { return Backend::AVX2; }
  bool Available() const override;
  PerfHints GetPerfHints() const override {
    PerfHints h;
    h.expected_latency_us_per_query_at_1e7 = 35000;  // ~5.7x faster than CPU
    h.expected_throughput_qps_at_1e7 = 28;
    h.has_simd = true;
    h.note = "AVX2 SIMD (256-bit)";
    return h;
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_AVX2_BACKEND_H_
