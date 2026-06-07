/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::simple_pir::SimplePirProtocol — algorithm-level static
 * functions for the SimplePIR protocol. First per-operator port chunk
 * built on top of the pir_core shared-infra layer
 * (Matrix + LweParams + GaussianSampler + Database).
 *
 * What this file implements (matches upstream simplepir's
 * pir/simple_pir.go method by method):
 *   * Init(params)     -> generates the public matrix A of shape
 *                         (M, N) sampled uniformly from Z_q. This is
 *                         the shared randomness every client uses to
 *                         construct queries against this database.
 *   * Setup(db, A, params) -> computes the per-database hint
 *                             H = DB.data * A and shifts DB to [0, p].
 *                             Returns H so the server can publish it
 *                             once and amortize across all clients;
 *                             the shifted DB stays mutated in place
 *                             so the Answer path reads from [0, p].
 *   * GenSecret(params)     -> uniform random N x 1 secret vector in
 *                              Z_q. Despite the upstream naming, the
 *                              secret itself is sampled UNIFORMLY
 *                              (matches simplepir's MatrixRand); the
 *                              Gaussian distribution applies to the
 *                              error vector inside Query, not to the
 *                              secret.
 *   * Query(i, A, s, p)     -> encrypted index lookup:
 *                              q = A*s + e + Delta * e_{i mod m}
 *                              where `e` is M x 1 Gaussian noise from
 *                              pir_core::GaussianSampler driven by a
 *                              caller-supplied mt19937_64. Squishing
 *                              padding (upstream's AppendZeros when
 *                              p.M % info.Squishing != 0) is skipped
 *                              this revision: DBinfo.squishing is 0
 *                              until Database::Squish/Unsquish lands.
 *
 * Intentionally deferred to follow-up commits:
 *   * Answer     — server-side compute over the squished DB. Needs
 *                  Squish/Unsquish in pir_core::Database first.
 *   * Recover    — client-side decoding with the hint. Symmetric to
 *                  Query; lands after Answer.
 *   * FakeSetup  — benchmark helper for hint-less timing runs.
 *   * Compressed seed paths — PRG-driven A regeneration to cut the
 *                             offline traffic. Lands after Init/Setup
 *                             prove correct.
 *
 * Activation: Setup and Query call Matrix::Mul which needs
 * PIR_PIR_CORE_REAL. Init and GenSecret are pure-arithmetic and run
 * in both stub and vendored modes. The kernel-calling methods forward
 * retcode::FAIL with caller-actionable `err` messages when the kernel
 * link is not vendored.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_PROTOCOL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_PROTOCOL_H_

#include <cstdint>
#include <random>
#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/gaussian.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::simple_pir {

class SimplePirProtocol {
 public:
  // Generates A in Z_q ^ (M x N). Uses Matrix::UniformRandom which is
  // pure C++ <random> seeded from std::random_device — fine for
  // testing and Setup correctness but NOT cryptographically strong;
  // production callers will swap to a CSPRNG-backed factory once it
  // exists (parallel of the GaussianSampler RNG-injection pattern).
  //
  // Parameters: `params` must have n, m, and logq populated (call
  // LweParams::Pick + ApproxSquareDatabaseDims first).
  //
  // Output: `A_out` is overwritten with a fresh M x N matrix.
  static retcode Init(const core::LweParams& params,
                      core::Matrix* A_out,
                      std::string* err);

  // Computes H = DB.data * A and shifts DB into [0, p] in place.
  //
  // Pre: db.data must be a (params.l x params.m) matrix sized to
  // match A's row count. `A` must be Init's output for THIS database
  // (caller pairs them).
  //
  // Post: H_out is the L x N hint matrix the server publishes. The
  // db's data is shifted by +p/2 so subsequent Answer reads see Z_p
  // values in [0, p) instead of the centered representation used
  // during Setup. This matches upstream's `DB.Data.Add(p/2)`.
  //
  // Activation: vendored mode required (calls Matrix::Mul which
  // forwards to @simplepir matMul). Stub mode returns FAIL with the
  // activation-flag hint.
  static retcode Setup(core::Database* db, const core::Matrix& A,
                       const core::LweParams& params,
                       core::Matrix* H_out,
                       std::string* err);

  // Generates an N x 1 secret vector uniform in Z_q. `params.n` and
  // `params.logq` must be populated. Output is written to `secret_out`.
  //
  // The secret is sampled with Matrix::UniformRandom (mt19937_64 +
  // random_device) which IS NOT cryptographically strong. Production
  // callers must inject a CSPRNG-backed factory before claiming LWE
  // security, same caveat as Init.
  static retcode GenSecret(const core::LweParams& params,
                           core::Matrix* secret_out,
                           std::string* err);

  // Builds the encrypted query vector for retrieving DB entry at index
  // `index`. The math (matches upstream simplepir's Query):
  //
  //   query = A * secret + e + Delta * e_{index mod m}
  //
  // where `e` is M x 1 Gaussian noise sampled via GaussianSampler
  // driven from `noise_rng`, and `e_k` is the standard basis vector
  // with a 1 at position k.
  //
  // Pre: A is (m x n), `secret` is (n x 1), params populated by Pick
  // and ApproxSquareDatabaseDims. params.sigma must equal the embedded
  // GaussianSampler sigma (currently 6.4); the check is enforced to
  // catch param-table / RNG-table drift early.
  //
  // Post: `query_out` is overwritten with an M x 1 ciphertext vector.
  // Squishing padding (upstream's `AppendZeros(info.Squishing - ...)`)
  // is NOT applied — the current Database leaves squishing=0, and the
  // un-squished Answer path expects an M x 1 query.
  //
  // RNG: `noise_rng` is caller-owned. Tests seed it deterministically;
  // production callers should pass a CSPRNG-backed instance for the
  // LWE noise. Same caveat applies as the Init / GenSecret RNG.
  //
  // Activation: vendored mode required (Matrix::Mul). Stub mode
  // forwards retcode::FAIL with the activation-flag hint.
  static retcode Query(uint64_t index,
                       const core::Matrix& A,
                       const core::Matrix& secret,
                       const core::LweParams& params,
                       std::mt19937_64* noise_rng,
                       core::Matrix* query_out,
                       std::string* err);
};

}  // namespace primihub::pir::simple_pir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_SIMPLE_PIR_PROTOCOL_H_
