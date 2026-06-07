/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SimplePirProtocol::Init / Setup / GenSecret / Query implementation.
 */
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_protocol.h"

#include <cstdint>
#include <random>
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

retcode SimplePirProtocol::GenSecret(const core::LweParams& params,
                                     core::Matrix* secret_out,
                                     std::string* err) {
  if (secret_out == nullptr) {
    if (err) *err = "SimplePirProtocol::GenSecret: secret_out is null";
    return retcode::FAIL;
  }
  if (params.n == 0 || params.logq == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::GenSecret: params not initialized — "
          << "n=" << params.n << " logq=" << params.logq
          << ". Call LweParams::Pick first.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // secret in Z_q ^ (N x 1). Matches upstream's
  //   secret := MatrixRand(p.N, 1, p.Logq, 0)
  // i.e. uniformly random — the Gaussian distribution applies to the
  // noise vector inside Query, not to the secret.
  *secret_out = core::Matrix::UniformRandom(params.n, 1,
                                            static_cast<uint32_t>(params.logq));
  return retcode::SUCCESS;
}

retcode SimplePirProtocol::Query(uint64_t index,
                                 const core::Matrix& A,
                                 const core::Matrix& secret,
                                 const core::LweParams& params,
                                 std::mt19937_64* noise_rng,
                                 core::Matrix* query_out,
                                 std::string* err) {
  if (query_out == nullptr || noise_rng == nullptr) {
    if (err) {
      *err = "SimplePirProtocol::Query: query_out or noise_rng is null";
    }
    return retcode::FAIL;
  }
  if (params.n == 0 || params.m == 0 || params.p == 0 ||
      params.logq == 0) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Query: params not initialized — "
          << "n=" << params.n << " m=" << params.m
          << " p=" << params.p << " logq=" << params.logq;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (A.rows() != params.m || A.cols() != params.n) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Query: A shape mismatch — A is "
          << A.rows() << "x" << A.cols()
          << ", expected " << params.m << "x" << params.n
          << " (params.m x params.n).";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (secret.rows() != params.n || secret.cols() != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Query: secret shape mismatch — secret is "
          << secret.rows() << "x" << secret.cols()
          << ", expected " << params.n << "x1 (params.n x 1).";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Pin the embedded GaussianSampler sigma to params.sigma. The lookup
  // table in lwe_params.cc and the CDF in gaussian.cc are independently
  // pinned to upstream simplepir@e9020b03; if a future upstream bump
  // changes sigma in one place and not the other, this check fires
  // before any noise is sampled.
  if (params.sigma != core::kGaussianSigma) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Query: params.sigma=" << params.sigma
          << " does not match embedded GaussianSampler sigma="
          << core::kGaussianSigma
          << ". Re-pin both kLweParamEntries[] and kGaussianCdfTable[]"
          << " from the same upstream commit.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  // query = A * secret  (m x 1) — kernel call. Stub mode returns FAIL
  // with the activation-flag hint, which propagates untouched.
  auto rc = A.Mul(secret, query_out, err);
  if (rc != retcode::SUCCESS) {
    return rc;
  }

  // Add Gaussian error. Sampler returns int64 in approximately
  // [-128, 128]; casting to uint32 is modular two's complement which
  // matches upstream MatrixAdd over C.Elem (= uint32).
  core::GaussianSampler sampler(*noise_rng);
  for (uint64_t i = 0; i < params.m; ++i) {
    const int64_t e = sampler.Sample();
    const uint32_t curr = query_out->Get(i, 0);
    query_out->Set(i, 0, curr + static_cast<uint32_t>(e));
  }

  // Encode the index by adding Delta() at position index mod m.
  const uint64_t k = index % params.m;
  const uint32_t curr_k = query_out->Get(k, 0);
  query_out->Set(k, 0,
                 curr_k + static_cast<uint32_t>(params.Delta()));

  // NOTE: Squishing padding (upstream's AppendZeros) intentionally
  // skipped — Database::Squish is DoublePIR scope and DBinfo.squishing
  // is 0 in this revision. When Squish lands, Query will need DBinfo
  // and conditional padding; see openspec tasks 5.5 / 7.2.
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
