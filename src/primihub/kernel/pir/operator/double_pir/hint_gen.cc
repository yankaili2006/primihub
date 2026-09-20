/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"

#include <chrono>
#include <string>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/double_pir/double_pir_protocol.h"

namespace primihub::pir::double_pir {

retcode HintGen::Compute(core::Database* db,
                         const core::LweParams& params,
                         DoublePirHint* hint_out,
                         std::string* err,
                         HintGenStats* stats_out) {
  using clock = std::chrono::steady_clock;
  using ms_d = std::chrono::duration<double, std::milli>;
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (db == nullptr || hint_out == nullptr) {
    return fail("HintGen::Compute: db / hint_out must be non-null");
  }
  if (params.p == 0 || params.logq == 0 || params.l == 0 ||
      params.m == 0 || params.n == 0) {
    return fail(
        "HintGen::Compute: params not fully populated (call "
        "LweParams::Pick + ApproxSquareDatabaseDims first)");
  }
  if (db->info().x == 0 || params.l % db->info().x != 0) {
    return fail(
        "HintGen::Compute: db.info().x must be nonzero and divide "
        "params.l (Database::SetupShape sets x = info.ne)");
  }

  // Init — sample the two public matrices.
  const auto t_init_start = clock::now();
  std::string init_err;
  if (DoublePirProtocol::Init(params, db->info(), &hint_out->A1,
                               &hint_out->A2, &init_err) != retcode::SUCCESS) {
    return fail("HintGen::Compute: Init failed: " + init_err);
  }
  const auto t_init_end = clock::now();

  // Setup — produce server states + per-database public hint.
  std::string setup_err;
  if (DoublePirProtocol::Setup(db, hint_out->A1, hint_out->A2, params,
                                &hint_out->H1_squished,
                                &hint_out->A2_copy_transposed,
                                &hint_out->H2_msg,
                                &setup_err) != retcode::SUCCESS) {
    return fail("HintGen::Compute: Setup failed: " + setup_err);
  }
  const auto t_setup_end = clock::now();
  if (stats_out != nullptr) {
    stats_out->init_ms = ms_d(t_init_end - t_init_start).count();
    stats_out->setup_ms = ms_d(t_setup_end - t_init_end).count();
  }

  // Capture the post-Setup DBinfo (basis/squishing/cols are populated
  // by Database::Squish inside Setup). Query+Answer+Recover need this.
  hint_out->info_after_setup = db->info();
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::double_pir
