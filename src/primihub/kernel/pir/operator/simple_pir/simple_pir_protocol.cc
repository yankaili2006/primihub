/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirProtocol::Init / Setup implementation.
 */
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_protocol.h"

#include <cstdint>
#include <sstream>
#include <string>

#include <glog/logging.h>

namespace primihub::pir::simple_pir {

retcode SimplePirProtocol::Init(const core::LweParams& params,
                                core::Matrix* A_out,
                                std::string* err) {
  if (A_out == nullptr) {
    if (err) *err = "SimplePirProtocol::Init: A_out is null";
    return retcode::FAIL;
  }
  if (params.n == 0 || params.m == 0 || params.logq == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Init: params not initialized — "
          << "n=" << params.n << " m=" << params.m
          << " logq=" << params.logq
          << ". Call LweParams::Pick + ApproxSquareDatabaseDims first.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // A in Z_q ^ (M x N). UniformRandom takes logmod -> [0, 2^logmod).
  // For SimplePIR's standard table logq=32 means full uint32 range,
  // which matches upstream's MatrixRand(p.M, p.N, p.Logq, 0).
  *A_out = core::Matrix::UniformRandom(params.m, params.n,
                                       static_cast<uint32_t>(params.logq));
  return retcode::SUCCESS;
}

retcode SimplePirProtocol::Setup(core::Database* db, const core::Matrix& A,
                                 const core::LweParams& params,
                                 core::Matrix* H_out,
                                 std::string* err) {
  if (db == nullptr || H_out == nullptr) {
    if (err) {
      *err = "SimplePirProtocol::Setup: db or H_out is null";
    }
    return retcode::FAIL;
  }
  if (params.p == 0 || params.logq == 0 || params.l == 0 ||
      params.m == 0 || params.n == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Setup: params not initialized — "
          << "n=" << params.n << " l=" << params.l
          << " m=" << params.m << " logq=" << params.logq
          << " p=" << params.p;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (db->data().rows() != params.l || db->data().cols() != params.m) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Setup: DB shape mismatch — db is "
          << db->data().rows() << "x" << db->data().cols()
          << ", params expect " << params.l << "x" << params.m
          << ". Caller must SetupShape / MakeRandom with the same "
          << "LweParams used here.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (A.rows() != params.m || A.cols() != params.n) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Setup: A shape mismatch — A is "
          << A.rows() << "x" << A.cols()
          << ", expected " << params.m << "x" << params.n
          << " (params.m x params.n).";
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  // H = DB.data * A. Matrix::Mul forwards to @simplepir matMul under
  // PIR_PIR_CORE_REAL; in stub mode it returns FAIL with the
  // activation-flag hint, which propagates through here untouched.
  auto rc = db->data().Mul(A, H_out, err);
  if (rc != retcode::SUCCESS) {
    return rc;
  }

  // Map DB entries from [-p/2, p/2] back to [0, p). Upstream does the
  // equivalent via Matrix::Add(p/2). We use ScalarAdd which is the
  // same wrapping uint32 arithmetic — Squish would normally happen
  // next; deferred to DoublePIR scope.
  db->mutable_data().ScalarAdd(static_cast<uint32_t>(params.p / 2));
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
