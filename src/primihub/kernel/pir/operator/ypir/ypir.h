/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// YPIR (Menon & Wu, USENIX'24) — single-server PIR with the lowest
// communication of any production-grade scheme to date. Same authors
// as SpiralPIR but cuts online query traffic from ~14 KB to under 1 KB
// at the cost of a small per-database hint. Picked by Selector when
// bandwidth_priority is set. Skeleton phase.
class YpirOperator : public BasePirOperator {
 public:
  explicit YpirOperator(const Options& options) : BasePirOperator(options) {}
  ~YpirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = true;
};

}  // namespace primihub::pir
#endif
