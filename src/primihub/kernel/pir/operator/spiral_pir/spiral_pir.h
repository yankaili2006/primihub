/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_PIR_H_

#include <string>
#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// SpiralPIR — single-server FHE-composition PIR (Menon & Wu, USENIX'22).
//
// Properties recorded in capabilities and verified by the Selector:
//   * single server, no per-database preprocessing
//   * semi-honest security against a single passive adversary
//   * query latency on the Seconds class at 1e8 rows (paper: ~2-3 s server time)
//   * recommended scale up to ~1e9 rows
//   * CPU backend only at present; AVX2 / CUDA backends planned (P7)
//
// IMPLEMENTATION STATUS — Phase 3 skeleton only:
//   This translation unit currently registers the algorithm and its
//   capabilities so that PirSelector::Recommend exposes spiral as an
//   option, the multi-algo framework lists it via `pir_inspect list`,
//   and any task plan can be authored against it. OnExecute returns
//   retcode::FAIL with a clear error log; the real query path will be
//   implemented in a follow-up commit once thirdparty/pir/BUILD.spiral
//   vendors menonsamir/spiral.
//
//   Callers asking PirRegistry::Create("spiral", opts) WILL get a
//   constructed operator instance; Execute will fail loudly rather
//   than silently produce wrong results. This is deliberate: the
//   selector / capabilities / registration plumbing is real, only
//   the cryptographic core is missing.
class SpiralPirOperator : public BasePirOperator {
 public:
  explicit SpiralPirOperator(const Options& options) : BasePirOperator(options) {}
  ~SpiralPirOperator() override = default;

  retcode OnExecute(const PirDataType& input, PirDataType* result) override;

  // Marker used by pir_inspect and (later) integration tests so that the
  // skeleton-vs-real distinction is observable at runtime without parsing
  // log lines. Stays true until the real implementation lands.
  static constexpr bool kIsSkeleton = true;
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_PIR_H_
