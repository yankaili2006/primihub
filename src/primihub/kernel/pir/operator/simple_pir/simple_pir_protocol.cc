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
                                 const core::DBinfo& info,
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

  // Squishing padding: when the DB has been Squished and params.m is
  // not divisible by info.squishing, append zero rows so the packed
  // Answer kernel can read (info.cols * info.squishing) elements.
  // When info.squishing == 0 (no Squish), this is a no-op and query
  // stays at native (m x 1) shape.
  if (info.squishing > 0 && (params.m % info.squishing) != 0) {
    const uint64_t pad = info.squishing - (params.m % info.squishing);
    const uint64_t new_rows = params.m + pad;
    core::Matrix padded = core::Matrix::Zeros(new_rows, 1);
    for (uint64_t i = 0; i < params.m; ++i) {
      padded.Set(i, 0, query_out->Get(i, 0));
    }
    *query_out = std::move(padded);
  }
  return retcode::SUCCESS;
}

retcode SimplePirProtocol::Answer(const core::Database& squished_db,
                                  const core::Matrix& query,
                                  core::Matrix* answer_out,
                                  std::string* err) {
  if (answer_out == nullptr) {
    if (err) *err = "SimplePirProtocol::Answer: answer_out is null";
    return retcode::FAIL;
  }
  const auto& info = squished_db.info();
  if (info.squishing == 0 || info.basis == 0) {
    if (err) {
      *err =
          "SimplePirProtocol::Answer: db not squished (info.squishing"
          " == 0). Call Database::Squish before Answer.";
    }
    return retcode::FAIL;
  }
  // Query must equal (squished_cols * squishing) — MulVecPacked guards
  // the same condition but we surface a SimplePIR-flavored error.
  const uint64_t expected = squished_db.data().cols() * info.squishing;
  if (query.cols() != 1 || query.rows() != expected) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Answer: query shape mismatch — got "
          << query.rows() << "x" << query.cols()
          << ", expected " << expected
          << "x1 (squished_db.cols=" << squished_db.data().cols()
          << " * squishing=" << info.squishing
          << "). Caller must pass the squishing-padded query Query"
          << " produced for the same Database.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  return squished_db.data().MulVecPacked(query, info.basis,
                                         info.squishing, answer_out,
                                         err);
}

retcode SimplePirProtocol::Recover(uint64_t index,
                                   const core::Matrix& query,
                                   const core::Matrix& hint,
                                   const core::Matrix& secret,
                                   const core::Matrix& answer,
                                   const core::LweParams& params,
                                   const core::DBinfo& info,
                                   uint64_t* recovered_out,
                                   std::string* err) {
  if (recovered_out == nullptr) {
    if (err) *err = "SimplePirProtocol::Recover: recovered_out is null";
    return retcode::FAIL;
  }
  if (params.p == 0 || params.m == 0 || params.n == 0 ||
      params.logq == 0 || params.l == 0) {
    if (err) *err = "SimplePirProtocol::Recover: params not initialized";
    return retcode::FAIL;
  }
  if (info.ne == 0 || info.p == 0 || info.logq == 0) {
    if (err) *err = "SimplePirProtocol::Recover: info not initialized";
    return retcode::FAIL;
  }
  if (hint.rows() != params.l || hint.cols() != params.n) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: hint shape mismatch — got "
          << hint.rows() << "x" << hint.cols()
          << ", expected " << params.l << "x" << params.n;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (secret.rows() != params.n || secret.cols() != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: secret shape mismatch — got "
          << secret.rows() << "x" << secret.cols()
          << ", expected " << params.n << "x1";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (answer.rows() != params.l || answer.cols() != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: answer shape mismatch — got "
          << answer.rows() << "x" << answer.cols()
          << ", expected " << params.l << "x1";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (query.cols() != 1 || query.rows() < params.m) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: query shape mismatch — got "
          << query.rows() << "x" << query.cols()
          << ", expected at least " << params.m << "x1 (Recover only"
          << " reads the first params.m entries; the rest are"
          << " squishing padding zeros).";
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  // Compute offset = -sum(query[j] * p/2) mod 2^logq. Upstream
  // mirrors this exact formula. Use uint64 with explicit mod for the
  // shift — params.logq is 32 in the current table; the formula
  // handles up to logq=63 via (uint64_t{1} << logq).
  const uint64_t q = uint64_t{1} << params.logq;
  const uint64_t ratio = params.p / 2;
  uint64_t offset = 0;
  for (uint64_t j = 0; j < params.m; ++j) {
    offset = (offset + ratio * query.Get(j, 0)) % q;
  }
  offset = q - offset;

  // interm = H * secret  (L x 1)
  core::Matrix interm;
  auto rc = hint.Mul(secret, &interm, err);
  if (rc != retcode::SUCCESS) return rc;
  if (interm.rows() != params.l || interm.cols() != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: interm shape unexpected — got "
          << interm.rows() << "x" << interm.cols();
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  // For SimplePIR, the index encodes which DB row was queried; row =
  // index / params.m, and within that row we decode `info.ne` Z_p
  // base-p digits.
  const uint64_t row = index / params.m;
  if ((row + 1) * info.ne > params.l) {
    if (err) {
      std::ostringstream oss;
      oss << "SimplePirProtocol::Recover: row=" << row
          << " ne=" << info.ne << " runs past l=" << params.l;
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  std::vector<uint64_t> vals;
  vals.reserve(info.ne);
  for (uint64_t j = row * info.ne; j < (row + 1) * info.ne; ++j) {
    // noised = answer[j] - interm[j] + offset, all mod 2^logq.
    const uint64_t answered = answer.Get(j, 0);
    const uint64_t interm_j = interm.Get(j, 0);
    // Subtract interm and add offset under modular arithmetic. Use
    // uint64 then reduce by q to avoid signedness pitfalls.
    uint64_t noised = (answered + q - (interm_j % q)) % q;
    noised = (noised + offset) % q;
    vals.push_back(params.Round(noised));
  }
  *recovered_out = core::ReconstructElem(std::move(vals), index, info);
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
