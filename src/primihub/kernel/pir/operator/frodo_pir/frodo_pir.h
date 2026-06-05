/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// FrodoPIR (de Castro & Lee, PETS'23) — single-server LWE-based PIR with
// per-database preprocessing. Industrially deployed by Brave for the
// Privacy-Preserving STAR analytics pipeline; the maturity (vs SimplePIR
// which is a research artefact) is the main reason an operator would pick
// it. Skeleton phase — see openspec tasks 6.x for the real impl.
class FrodoPirOperator : public BasePirOperator {
 public:
  explicit FrodoPirOperator(const Options& options) : BasePirOperator(options) {}
  ~FrodoPirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = true;
};

}  // namespace primihub::pir
#endif
