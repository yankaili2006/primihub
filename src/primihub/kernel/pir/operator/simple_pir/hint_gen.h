/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::simple_pir::SimpleHintGen — extracts SimplePIR's
 * Init + Setup + Squish into a reusable static function so the same
 * offline work can be (a) cached in-process, (b) persisted to disk,
 * and (c) shared across operators that hold the same DB.
 *
 * Mirrors the DoublePIR HintGen abstraction from task 5.6 chunk 1
 * (commit 7303a83e). Same shape, different hint payload (SimplePIR's
 * hint is just `A` + `H`, both single matrices, vs DoublePIR's
 * 5-matrix bundle).
 *
 * SimplePIR's offline cost profile: Init samples a single l×n public
 * matrix; Setup multiplies DB·A producing the l×n hint H; Squish
 * compresses DB in place. The dominant cost is Setup (O(L·M·n)).
 * Caching wins are identical in shape to DoublePIR's — at 1e8 entries
 * SimplePIR's hint is the largest in the framework (~1.2 GB per the
 * paper's Table 3), so persistence amortization matters even more.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_GEN_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_GEN_H_

#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::simple_pir {

// All state produced by Init + Setup + Squish that the query loop needs.
//
// A is the public matrix shared with the client; H is the precomputed
// DB·A hint also shared with the client. info_after_squish carries
// basis/squishing/cols populated by Database::Squish so downstream
// Answer / Recover have the right shape metadata.
//
// secret is intentionally NOT part of the hint — it's per-client /
// per-session randomness sampled fresh in OnExecute.
struct SimplePirHint {
  core::Matrix A;
  core::Matrix H;
  core::DBinfo info_after_squish;
};

// Optional per-stage timing. Populated when caller passes a non-null
// SimpleHintGenStats* — same opt-in pattern as DoublePIR's HintGenStats.
struct SimpleHintGenStats {
  double init_ms = 0.0;
  double setup_ms = 0.0;
  double squish_ms = 0.0;
};

class SimpleHintGen {
 public:
  // Computes the offline hint for `*db` and squishes db in place so it
  // matches the post-Setup shape Answer expects.
  //
  // Pre: params populated by LweParams::Pick(doublepir=false, …) +
  //      ApproxSquareDatabaseDims; db.SetupShape() already called +
  //      bytes loaded + shifted to centered [-p/2, p/2) by the caller
  //      (Setup itself does NOT re-shift — it does `db.Data.Add(p/2)`
  //      to undo).
  // Post on SUCCESS: hint_out populated; db is now squished (basis/
  //      squishing/cols populated in db.info()).
  //
  // Stub-mode (no vendored kernels) returns retcode::FAIL with err
  // populated mentioning the activation flag.
  static retcode Compute(core::Database* db,
                         const core::LweParams& params,
                         SimplePirHint* hint_out,
                         std::string* err,
                         SimpleHintGenStats* stats_out = nullptr);

  // Squish basis / factor are hard-coded the same way the upstream
  // simplepir paper picks them. Exposed as constants so the cache
  // re-Squish path (after a hit) uses identical values.
  static constexpr uint64_t kSquishBasis = 10;
  static constexpr uint64_t kSquishingFactor = 3;
};

}  // namespace primihub::pir::simple_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_GEN_H_
