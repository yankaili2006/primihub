/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CUDA_BACKEND_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CUDA_BACKEND_H_

#include "src/primihub/kernel/pir/operator/backend/backend.h"

namespace primihub::pir {

class CudaBackend : public PirBackend {
 public:
  std::string Name() const override { return "cuda"; }
  Backend Type() const override { return Backend::CUDA; }
  bool Available() const override;
  PerfHints GetPerfHints() const override {
    PerfHints h;
    h.expected_latency_us_per_query_at_1e7 = 8000;  // ~25x faster than CPU
    h.expected_throughput_qps_at_1e7 = 125;
    h.has_simd = true;
    h.note = "NVIDIA CUDA";
    return h;
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_BACKEND_CUDA_BACKEND_H_
