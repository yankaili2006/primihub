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
//
// OnExecute interface (single-process self-contained retrieval — the
// production client/server split is a separate transport-layer chunk):
//   input["db_content"]    — vector of N decimal byte strings in [0, 255]
//   input["query_indices"] — vector of K decimal index strings in [0, N)
// On success, sets result["recovered"] to a vector of K decimal byte
// strings, in order.
//
// Constraints: N must yield l = floor(sqrt(NumDbEntries(N, 8, p))) that
// is a multiple of 8 (matMulVecPacked processes rows in batches of 8 and
// reads OOB otherwise). Smallest legal N is 64 → l=8. Requires the
// pir_core vendored mode (--define=enable_pir_core_real=1 +
// @simplepir override) — stub mode returns FAIL with the activation
// flag hint.
class SimplePirOperator : public BasePirOperator {
 public:
  explicit SimplePirOperator(const Options& options) : BasePirOperator(options) {}
  ~SimplePirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  // OnExecute drives the full SimplePirProtocol pipeline in vendored
  // mode (Init -> Setup -> Squish -> GenSecret -> Query -> Answer ->
  // Recover). kIsSkeleton stays false; stub mode gracefully FAILs.
  static constexpr bool kIsSkeleton = false;
};

}  // namespace primihub::pir
#endif
