/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/simple_pir/hint_gen.h"

#include <chrono>
#include <string>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_protocol.h"

namespace primihub::pir::simple_pir {

retcode SimpleHintGen::Compute(core::Database* db,
                               const core::LweParams& params,
                               SimplePirHint* hint_out,
                               std::string* err,
                               SimpleHintGenStats* stats_out) {
  using clock = std::chrono::steady_clock;
  using ms_d = std::chrono::duration<double, std::milli>;
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (db == nullptr || hint_out == nullptr) {
    return fail("SimpleHintGen::Compute: db / hint_out must be non-null");
  }
  if (params.p == 0 || params.logq == 0 || params.l == 0 ||
      params.m == 0 || params.n == 0) {
    return fail(
        "SimpleHintGen::Compute: params not fully populated (call "
        "LweParams::Pick + ApproxSquareDatabaseDims first)");
  }

  // Init — sample public matrix A.
  const auto t_init_start = clock::now();
  std::string init_err;
  if (SimplePirProtocol::Init(params, &hint_out->A, &init_err) !=
      retcode::SUCCESS) {
    return fail("SimpleHintGen::Compute: Init failed: " + init_err);
  }
  const auto t_init_end = clock::now();

  // Setup — compute H = DB · A, in-place +p/2 shift on db.
  std::string setup_err;
  if (SimplePirProtocol::Setup(db, hint_out->A, params, &hint_out->H,
                                &setup_err) != retcode::SUCCESS) {
    return fail("SimpleHintGen::Compute: Setup failed: " + setup_err);
  }
  const auto t_setup_end = clock::now();

  // Squish — in-place DB compression. Must run before Answer to match
  // the squished-db shape MulVecPacked expects.
  std::string sq_err;
  if (db->Squish(kSquishBasis, kSquishingFactor, &sq_err) !=
      retcode::SUCCESS) {
    return fail("SimpleHintGen::Compute: Database::Squish failed: " + sq_err);
  }
  const auto t_squish_end = clock::now();

  hint_out->info_after_squish = db->info();

  if (stats_out != nullptr) {
    stats_out->init_ms = ms_d(t_init_end - t_init_start).count();
    stats_out->setup_ms = ms_d(t_setup_end - t_init_end).count();
    stats_out->squish_ms = ms_d(t_squish_end - t_setup_end).count();
  }
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
