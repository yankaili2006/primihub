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
// per-database hint; the online answer is a SimplePIR linear answer plus a
// BFV homomorphic inner-product, decrypted client-side. Selected for
// Semantic query types.
//
// v1 (openspec change primihub-pir-cuda-tiptoe, task 1.1) ports the core
// LHE-on-SimplePIR retrieval as a PirOperator. The crypto core is the
// already-C++ `ahenzinger/underhood` rlwe layer over Microsoft SEAL; the LHE
// protocol (client/server/hint/params/secret) is ported from underhood's Go;
// the linear layer reuses primihub's ported SimplePIR core. The search/
// application layer (embeddings, k-means clustering, the 2-round coordinator)
// is deferred. See docs/pir/tiptoe-port-plan.md.
//
// SKELETON: OnExecute is not yet implemented (kIsSkeleton=true,
// caps.is_real=false). The real cryptographic core lands in chunks 1.1b-1.1f.
class TiptoePirOperator : public BasePirOperator {
 public:
  explicit TiptoePirOperator(const Options& options)
      : BasePirOperator(options) {}
  ~TiptoePirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = true;
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_PIR_H_
