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
// (ypir_e2e_test). Scaled to the paper's poly_len=2048 expansion preset
// (ParamsForExpansion, t_exp_left=2, hamming=256) -- ypir_e2e_large_test and
// ypir_operator_test both retrieve exactly at 2048. The online answer still
// uses recursive ring packing (precompute_pack + AVX512 packing is the
// remaining sub-second-per-query speed follow-up).
class YpirOperator : public BasePirOperator {
 public:
  explicit YpirOperator(const Options& options) : BasePirOperator(options) {}
  ~YpirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = false;
};

}  // namespace primihub::pir
#endif
