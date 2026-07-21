/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// Tiptoe (Henzinger et al., USENIX/SOSP'23) — private nearest-neighbor /
// semantic search built as BFV-on-SimplePIR. The server holds an offline
// per-database hint; the online answer is a SimplePIR linear answer minus a
// BFV-recovered H*s correction, decoded client-side.
//
// v1 (openspec change primihub-pir-cuda-tiptoe, task 1.1) implements the core
// LHE-on-SimplePIR retrieval (tiptoe_lhe_pir.h): a ternary-secret SimplePIR
// linear layer + the LHE hint machinery (Enc(s) -> Enc(H*s) -> recover) over
// the vendored underhood/rlwe BFV crypto. The search/ application layer
// (embeddings, k-means clustering, 2-round coordinator) is deferred. See
// docs/pir/tiptoe-port-plan.md.
//
// The real query path needs Microsoft SEAL (GFW-blocked on .50), so it is gated
// behind --define=enable_tiptoe_real=1 (like APSI/keyword_pir). In the default
// build the operator is a skeleton (OnExecute returns FAIL); with the define it
// is a real operator validated end-to-end (exact byte retrieval).
class TiptoePirOperator : public BasePirOperator {
 public:
  explicit TiptoePirOperator(const Options& options)
      : BasePirOperator(options) {}
  ~TiptoePirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
#ifdef PIR_TIPTOE_RLWE_VENDORED
  static constexpr bool kIsSkeleton = false;
#else
  static constexpr bool kIsSkeleton = true;
#endif
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PIR_H_
