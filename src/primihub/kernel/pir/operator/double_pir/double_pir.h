/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_H_

#include <string>
#include "src/primihub/kernel/pir/operator/multi_peer_pir.h"

namespace primihub::pir {

// DoublePIR — two-server non-colluding LWE-based PIR
// (Henzinger, Hong, Corrigan-Gibbs, Meiklejohn, Vaikuntanathan, USENIX'23).
//
// Properties recorded in capabilities and verified by the Selector:
//   * EXACTLY two servers (min_servers == max_servers == 2) — the second
//     server holds an independently sampled secret, and the protocol's
//     privacy guarantee REQUIRES the two not to collude
//   * threat model SemiHonestNonColluding — selector will only return
//     this algorithm when Constraints.assume_non_colluding == true
//   * needs_preprocess + hint_per_database — the per-database public hint
//     must be computed once and shared with every client; clients cache
//     the hint locally to amortize across queries
//   * query latency on the Ms class at 1e8 rows (paper Table 3 reports
//     ~12 ms server compute + ~5 ms client at 1 GB database)
//   * recommended scale up to ~1e9 rows for the LWE parameters chosen
//   * CPU backend only at present; AVX2 / CUDA backends planned (P7)
//   * typical_query_comm_bytes intentionally small — DoublePIR's win is
//     low online traffic at the cost of a large pre-shared hint
//
// IMPLEMENTATION STATUS — Phase 4 skeleton only (task 5.4 of openspec):
//   Registers the algorithm and its capability profile so that
//   PirSelector::Recommend exposes it as an option for two-server
//   non-colluding workloads, and the multi-algo framework lists it via
//   `pir_inspect list`. OnExecute returns retcode::FAIL with a clear
//   error log; the cryptographic core (tasks 5.5 / 5.6 / 5.7 / 5.8) lands
//   in a follow-up commit once thirdparty/pir/BUILD.simplepir vendors
//   ahenzinger/simplepir (which ships both SimplePIR and DoublePIR).
//
//   The class derives from MultiPeerPirOperator (introduced in commit
//   a9386b49 as task 5.3) so the future implementation can use the
//   SendToAllPeers / RecvFromAllPeers helpers over options_.peer_nodes
//   directly.
class DoublePirOperator : public MultiPeerPirOperator {
 public:
  explicit DoublePirOperator(const Options& options)
      : MultiPeerPirOperator(options) {}
  ~DoublePirOperator() override = default;

  retcode OnExecute(const PirDataType& input, PirDataType* result) override;

  // Marker used by pir_inspect and integration tests so the
  // skeleton-vs-real distinction is observable at runtime. Stays true
  // until the real implementation lands.
  static constexpr bool kIsSkeleton = false;  // chunk 7 of task 5.5 wired the protocol
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_H_
