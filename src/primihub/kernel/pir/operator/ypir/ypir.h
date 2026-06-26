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
// bandwidth_priority is set.
//
// The full SimplePIR query path is now ported (task 7.3): OnExecute drives
// the offline hint + online answer (AVX512-dispatched packed-db kernel) +
// recursive ring packing + Regev decrypt, validated end-to-end
// (ypir_e2e_test). v1 uses the small poly_len=8 binary-gadget preset
// (single-byte elements); scaling to the paper's poly_len=2048 preset is a
// tuning follow-up.
class YpirOperator : public BasePirOperator {
 public:
  explicit YpirOperator(const Options& options) : BasePirOperator(options) {}
  ~YpirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = false;
};

}  // namespace primihub::pir
#endif
