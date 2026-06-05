/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// SimplePIR — single-server LWE-based PIR with cached client hint
// (Henzinger, Hong, Corrigan-Gibbs, Meiklejohn, Vaikuntanathan, USENIX'23,
// the single-server companion to DoublePIR in the same paper).
//
// SimplePIR keeps the threat model at plain SemiHonest (the second server
// of DoublePIR is what introduces the non-colluding assumption). Clients
// download a per-database hint once and amortize across many queries.
// Skeleton phase — see openspec tasks 5.4 / 5.5 for the real impl.
class SimplePirOperator : public BasePirOperator {
 public:
  explicit SimplePirOperator(const Options& options) : BasePirOperator(options) {}
  ~SimplePirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = true;
};

}  // namespace primihub::pir
#endif
