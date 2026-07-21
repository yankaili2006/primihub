/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::double_pir::HintGen — separates the DoublePIR offline
 * hint computation (Init + Setup) from the per-query Answer + Recover
 * loop. The split serves two long-term goals:
 *
 *   1. Caching. Hint generation is O(L·M·n) — dominated by the H1=DB·A1
 *      matrix multiply. For a 1e8 DB this is ~110 s. A query-time
 *      cache lookup keyed by (l, m, p, db fingerprint) lets repeat
 *      callers amortize the cost across many queries.
 *
 *   2. Two-server network split (task 5.6). In production DoublePIR
 *      runs across two non-colluding servers: Server-1 holds the
 *      database + the H1_squished hint, Server-2 holds A2_copy_transposed
 *      + the A2 public matrix's secret-key half. HintGen produces all
 *      three pieces in one place; a future LinkContext-aware sibling
 *      will compute them in parallel across two peers and serialize
 *      each peer's slice onto the wire.
 *
 * This first chunk lands ONLY the refactor — extract the Init+Setup
 * sequence currently inlined in DoublePirOperator::OnExecute into a
 * static HintGen::Compute. Caching + network split land as follow-up
 * chunks once the data flow is settled.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_GEN_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_GEN_H_

#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {

// All state produced by Init + Setup that the query loop needs.
//
// A1 / A2 are the public random matrices sampled by Init — both client
// and server need them. The client side uses them in Query (LWE encrypt);
// the server side uses them transitively (Setup folds them into H1 and
// H2_msg). info_after_setup carries the basis/squishing/cols populated
// by Database::Squish during Setup, which downstream Query/Recover
// need.
//
// Wire-protocol note: in a 2-server split, A1 + A2 + H2_msg are public
// (shareable with client); H1_squished is server-1's secret server
// state; A2_copy_transposed is server-2's secret server state. The
// struct keeps them together for the single-process case; a future
// HintBundle::Split() helper will partition them by role.
struct DoublePirHint {
  core::Matrix A1;
  core::Matrix A2;
  core::Matrix H1_squished;
  core::Matrix A2_copy_transposed;
  core::Matrix H2_msg;
  core::DBinfo info_after_setup;
};

// Per-stage timing captured by HintGen::Compute. Optional output —
// callers that don't pass a Stats pointer get the same behaviour as
// before. Used by DoublePirOperator to surface init_ms / setup_ms
// separately in its observability LOG line.
struct HintGenStats {
  double init_ms = 0.0;
  double setup_ms = 0.0;
};

class HintGen {
 public:
  // Compute the offline hint for `*db`. db must be pre-populated by the
  // caller — bytes loaded, centered representation applied. Setup
  // mutates db.info() in place (basis=10 / squishing=3 / cols updated)
  // and returns the resulting DBinfo in hint_out.info_after_setup.
  //
  // Pre: params populated by LweParams::Pick(doublepir=true, …) +
  //      ApproxSquareDatabaseDims; db.SetupShape() already called.
  // Post on SUCCESS: hint_out fields populated; db is now squished
  //      (caller should treat it as the server-1 database state).
  //
  // Activation: vendored mode required (Matrix::Mul / Transpose are
  // kernel calls). Stub mode forwards retcode::FAIL with the kernel-
  // bridge's activation-flag hint.
  // stats_out is optional — pass nullptr to skip timing capture.
  static retcode Compute(core::Database* db,
                         const core::LweParams& params,
                         DoublePirHint* hint_out,
                         std::string* err,
                         HintGenStats* stats_out = nullptr);
};

}  // namespace primihub::pir::double_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_GEN_H_
