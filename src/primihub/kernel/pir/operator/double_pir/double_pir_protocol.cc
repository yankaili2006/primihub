/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir_protocol.h"

#include "src/primihub/kernel/pir/operator/pir_core/gaussian.h"

#include <sstream>

#include <glog/logging.h>

namespace primihub::pir::double_pir {

retcode DoublePirProtocol::Init(const core::LweParams& params,
                                const core::DBinfo& info,
                                core::Matrix* A1_out,
                                core::Matrix* A2_out,
                                std::string* err) {
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (A1_out == nullptr || A2_out == nullptr) {
    return fail("DoublePirProtocol::Init: A1_out / A2_out must be non-null");
  }
  if (params.m == 0 || params.n == 0 || params.l == 0 || params.logq == 0) {
    return fail(
        "DoublePirProtocol::Init: params must have m / n / l / logq "
        "populated — call LweParams::Pick + ApproxSquareDatabaseDims "
        "first");
  }
  if (info.x == 0) {
    return fail("DoublePirProtocol::Init: info.x == 0");
  }
  if (params.l % info.x != 0) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Init: info.x=" << info.x
        << " must divide params.l=" << params.l
        << " — DoublePIR's compression ratio requires exact partition";
    return fail(oss.str());
  }
  // Upstream: A1 := MatrixRand(p.M, p.N, p.Logq, 0)
  //          A2 := MatrixRand(p.L / info.X, p.N, p.Logq, 0)
  // UniformRandom takes logmod and returns values in [0, 2^logmod),
  // matching the upstream MatrixRand(rows, cols, logq, 0) signature
  // (the trailing 0 there means "use bound = 2^logq" rather than a
  // custom modulus).
  *A1_out = core::Matrix::UniformRandom(params.m, params.n, params.logq);
  *A2_out = core::Matrix::UniformRandom(params.l / info.x, params.n,
                                         params.logq);
  return retcode::SUCCESS;
}

retcode DoublePirProtocol::Setup(core::Database* db,
                                 const core::Matrix& A1,
                                 const core::Matrix& A2,
                                 const core::LweParams& params,
                                 core::Matrix* H1_squished_out,
                                 core::Matrix* A2_copy_transposed_out,
                                 core::Matrix* H2_msg_out,
                                 std::string* err) {
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (db == nullptr || H1_squished_out == nullptr ||
      A2_copy_transposed_out == nullptr || H2_msg_out == nullptr) {
    return fail(
        "DoublePirProtocol::Setup: db / H1_squished_out / "
        "A2_copy_transposed_out / H2_msg_out must all be non-null");
  }
  if (params.p == 0 || params.logq == 0 || params.l == 0 ||
      params.m == 0 || params.n == 0) {
    return fail(
        "DoublePirProtocol::Setup: params must have p/logq/l/m/n "
        "populated (call LweParams::Pick first)");
  }
  const core::DBinfo& info = db->info();
  if (info.x == 0 || params.l % info.x != 0) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Setup: info.x=" << info.x
        << " must be nonzero and divide params.l=" << params.l;
    return fail(oss.str());
  }
  if (A1.rows() != params.m || A1.cols() != params.n) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Setup: A1 shape mismatch — expected "
        << params.m << "x" << params.n << ", got " << A1.rows() << "x"
        << A1.cols();
    return fail(oss.str());
  }
  if (A2.rows() != params.l / info.x || A2.cols() != params.n) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Setup: A2 shape mismatch — expected "
        << (params.l / info.x) << "x" << params.n << ", got "
        << A2.rows() << "x" << A2.cols();
    return fail(oss.str());
  }

  // ---- H1 = DB.data * A1 ----  shape L x N.
  core::Matrix H1;
  std::string mul_err;
  if (db->data().Mul(A1, &H1, &mul_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Setup: DB * A1 failed: " + mul_err);
  }

  // ---- H1.Transpose() ----  shape N x L.
  core::Matrix H1_t;
  std::string tr_err;
  if (H1.Transpose(&H1_t, &tr_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Setup: Transpose(H1) failed: " + tr_err);
  }
  H1 = std::move(H1_t);

  // ---- H1.Expand(params.p, params.NumBasePDigits()) ----  shape
  //      (N * delta) x L, where delta = ceil(logq / log2(p)).
  const uint64_t delta = params.NumBasePDigits();
  H1.Expand(params.p, delta);

  // ---- H1.ConcatCols(info.x) ----  shape (N * delta * X) x (L / X).
  H1.ConcatCols(info.x);

  // ---- H2 = H1 * A2 ----  shape (N * delta * X) x N.
  std::string h2_err;
  if (H1.Mul(A2, H2_msg_out, &h2_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Setup: H1 * A2 failed: " + h2_err);
  }

  // ---- Shift DB into [0, p] and Squish it (basis=10, squishing=3).
  // Upstream's DB.Data.Add(p.P/2) + DB.Squish() pair.
  db->mutable_data().ScalarAdd(static_cast<uint32_t>(params.p / 2));
  std::string sq_err;
  if (db->Squish(10, 3, &sq_err) != retcode::SUCCESS) {
    return fail(
        "DoublePirProtocol::Setup: db->Squish(10, 3) failed: " + sq_err);
  }

  // ---- Shift H1 into [0, p] and Squish it (same params).
  // Upstream's H1.Add(p.P/2) + H1.Squish(10, 3). After this, H1's
  // rows stay (N * delta * X); cols becomes ceil((L / X) / 3).
  H1.ScalarAdd(static_cast<uint32_t>(params.p / 2));
  H1.Squish(10, 3);
  *H1_squished_out = std::move(H1);

  // ---- A2_copy: deep copy, pad rows to a multiple of 3, transpose.
  // Upstream:
  //   A2_copy := A2.RowsDeepCopy(0, A2.Rows)  // full copy
  //   if A2_copy.Rows % 3 != 0:
  //       A2_copy.Concat(MatrixZeros(3 - (A2_copy.Rows % 3),
  //                                   A2_copy.Cols))
  //   A2_copy.Transpose()
  core::Matrix A2_copy = A2;  // value-copy via std::vector copy
  const uint64_t r = A2_copy.rows();
  if (r % 3 != 0) {
    const uint64_t pad = 3 - (r % 3);
    A2_copy.Concat(core::Matrix(pad, A2_copy.cols()));
  }
  std::string a2_tr_err;
  if (A2_copy.Transpose(A2_copy_transposed_out, &a2_tr_err) !=
      retcode::SUCCESS) {
    return fail(
        "DoublePirProtocol::Setup: Transpose(A2_copy) failed: " + a2_tr_err);
  }

  return retcode::SUCCESS;
}

retcode DoublePirProtocol::Query(
    uint64_t index, const core::Matrix& A1, const core::Matrix& A2,
    const core::LweParams& params, const core::DBinfo& info,
    std::mt19937_64* noise_rng, core::Matrix* secret1_out,
    core::Matrix* query1_out,
    std::vector<core::Matrix>* secrets2_out,
    std::vector<core::Matrix>* queries2_out, std::string* err) {
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (noise_rng == nullptr || secret1_out == nullptr ||
      query1_out == nullptr || secrets2_out == nullptr ||
      queries2_out == nullptr) {
    return fail("DoublePirProtocol::Query: out pointers / noise_rng must be non-null");
  }
  if (params.m == 0 || params.n == 0 || params.l == 0 ||
      params.p == 0 || params.logq == 0) {
    return fail(
        "DoublePirProtocol::Query: params must have m/n/l/p/logq populated");
  }
  if (info.x == 0 || info.ne == 0) {
    return fail(
        "DoublePirProtocol::Query: info.x / info.ne must be nonzero");
  }
  if (info.ne % info.x != 0) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Query: info.ne=" << info.ne
        << " must be divisible by info.x=" << info.x;
    return fail(oss.str());
  }
  if (params.l % info.x != 0) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Query: info.x=" << info.x
        << " must divide params.l=" << params.l;
    return fail(oss.str());
  }
  if (info.squishing == 0) {
    return fail(
        "DoublePirProtocol::Query: info.squishing must be populated "
        "(call DoublePirProtocol::Setup first which Squishes the DB)");
  }
  if (A1.rows() != params.m || A1.cols() != params.n) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Query: A1 shape mismatch — expected "
        << params.m << "x" << params.n << ", got " << A1.rows() << "x"
        << A1.cols();
    return fail(oss.str());
  }
  const uint64_t lx = params.l / info.x;
  if (A2.rows() != lx || A2.cols() != params.n) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Query: A2 shape mismatch — expected "
        << lx << "x" << params.n << ", got " << A2.rows() << "x"
        << A2.cols();
    return fail(oss.str());
  }

  const uint64_t i1 = (index / params.m) * (info.ne / info.x);
  const uint64_t i2 = index % params.m;

  // ---- secret1 + query1 (row index ciphertext).
  *secret1_out =
      core::Matrix::UniformRandom(params.n, 1, params.logq);
  std::string mul_err;
  if (A1.Mul(*secret1_out, query1_out, &mul_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Query: A1 * secret1 failed: " + mul_err);
  }
  // Add Gaussian noise vector (M x 1) and the Delta * e_{i2} bump.
  // GaussianSampler matches the SimplePIR Query pattern (see
  // simple_pir_protocol.cc); reads from the caller-owned mt19937_64
  // so tests can pin the noise.
  core::GaussianSampler sampler(*noise_rng);
  for (uint64_t i = 0; i < params.m; ++i) {
    const int64_t e = sampler.Sample();
    const uint32_t curr = query1_out->Get(i, 0);
    query1_out->Set(i, 0, curr + static_cast<uint32_t>(e));
  }
  const uint32_t curr_i2 = query1_out->Get(i2, 0);
  query1_out->Set(i2, 0,
                  curr_i2 + static_cast<uint32_t>(params.Delta()));
  // Pad to a multiple of info.squishing — matches upstream
  // AppendZeros for the row-query.
  if (params.m % info.squishing != 0) {
    query1_out->AppendZeros(info.squishing - (params.m % info.squishing));
  }

  // ---- secret2[j] + query2[j] (per-column ciphertexts).
  // info.ne / info.x distinct (secret, query) pairs; each pair has
  // its own LWE secret and noise vector.
  secrets2_out->clear();
  queries2_out->clear();
  const uint64_t num_pairs = info.ne / info.x;
  secrets2_out->reserve(num_pairs);
  queries2_out->reserve(num_pairs);
  for (uint64_t j = 0; j < num_pairs; ++j) {
    core::Matrix secret2 =
        core::Matrix::UniformRandom(params.n, 1, params.logq);
    core::Matrix query2;
    std::string m2_err;
    if (A2.Mul(secret2, &query2, &m2_err) != retcode::SUCCESS) {
      std::ostringstream oss;
      oss << "DoublePirProtocol::Query: A2 * secret2[" << j << "] failed: "
          << m2_err;
      return fail(oss.str());
    }
    for (uint64_t i = 0; i < lx; ++i) {
      const int64_t e = sampler.Sample();
      const uint32_t curr = query2.Get(i, 0);
      query2.Set(i, 0, curr + static_cast<uint32_t>(e));
    }
    const uint64_t k = i1 + j;
    if (k >= lx) {
      std::ostringstream oss;
      oss << "DoublePirProtocol::Query: index=" << index
          << " produced i1+j=" << k << " >= L/X=" << lx
          << " — index out of bounds for the database";
      return fail(oss.str());
    }
    const uint32_t curr_k = query2.Get(k, 0);
    query2.Set(k, 0, curr_k + static_cast<uint32_t>(params.Delta()));
    if (lx % info.squishing != 0) {
      query2.AppendZeros(info.squishing - (lx % info.squishing));
    }
    secrets2_out->push_back(std::move(secret2));
    queries2_out->push_back(std::move(query2));
  }
  return retcode::SUCCESS;
}

retcode DoublePirProtocol::Answer(
    const core::Database& squished_db, const core::Matrix& H1_squished,
    const core::Matrix& A2_copy_transposed, const core::Matrix& query1,
    const std::vector<core::Matrix>& queries2,
    const core::LweParams& params, core::Matrix* answer1_out,
    std::vector<core::Matrix>* answers2_out, std::string* err) {
  auto fail = [&](const std::string& msg) {
    if (err) *err = msg;
    return retcode::FAIL;
  };
  if (answer1_out == nullptr || answers2_out == nullptr) {
    return fail(
        "DoublePirProtocol::Answer: answer1_out / answers2_out must be "
        "non-null");
  }
  const core::DBinfo& info = squished_db.info();
  if (info.basis != 10 || info.squishing != 3) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Answer: squished_db must have basis=10 "
           "squishing=3 (got basis="
        << info.basis << " squishing=" << info.squishing
        << ") — call Setup first";
    return fail(oss.str());
  }
  if (info.x == 0 || info.ne % info.x != 0) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Answer: bad info — x=" << info.x
        << " ne=" << info.ne << " (need x != 0 and x | ne)";
    return fail(oss.str());
  }
  const uint64_t per_col = info.ne / info.x;
  if (queries2.size() != per_col) {
    std::ostringstream oss;
    oss << "DoublePirProtocol::Answer: expected " << per_col
        << " queries2 (= info.ne / info.x), got " << queries2.size();
    return fail(oss.str());
  }
  if (params.p == 0 || params.logq == 0) {
    return fail("DoublePirProtocol::Answer: params.p/logq must be populated");
  }

  // ---- a1 = squished_db * query1 ----  shape (L, 1).
  // Upstream: a := MatrixMulVecPacked(DB.Data, q1, basis, squishing)
  core::Matrix a1;
  std::string mvp_err;
  if (squished_db.data().MulVecPacked(query1, info.basis, info.squishing,
                                       &a1, &mvp_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Answer: squished_db * query1 failed: " +
                mvp_err);
  }

  // ---- TransposeAndExpandAndConcatColsAndSquish(p.P, p.delta(), info.X, 10, 3)
  // Composition: Transpose + Expand + ScalarAdd(p/2) + ConcatCols + Squish.
  // The ScalarAdd(p/2) un-does Matrix::Expand's centering subtraction
  // so the squished bit pattern matches upstream's un-centered fused op.
  core::Matrix a1_t;
  std::string tr_err;
  if (a1.Transpose(&a1_t, &tr_err) != retcode::SUCCESS) {
    return fail("DoublePirProtocol::Answer: Transpose(a1) failed: " + tr_err);
  }
  const uint64_t delta = params.NumBasePDigits();
  a1_t.Expand(params.p, delta);
  a1_t.ScalarAdd(static_cast<uint32_t>(params.p / 2));
  a1_t.ConcatCols(info.x);
  a1_t.Squish(10, 3);
  // a1_t now has shape (n * delta * x, ceil((l / x) / 3)).

  // ---- h1 = MulTransposedPacked(a1_t, A2_copy_transposed, 10, 3) ----
  std::string mtp_err;
  if (a1_t.MulTransposedPacked(A2_copy_transposed, 10, 3, answer1_out,
                                &mtp_err) != retcode::SUCCESS) {
    return fail(
        "DoublePirProtocol::Answer: MulTransposedPacked(a1, A2T) failed: " +
        mtp_err);
  }

  // ---- Per-query2 LWE: for j in [0, info.ne / info.x):
  //        a2 = H1_squished.MulVecPacked(q2)
  //        h2 = a1_t.MulVecPacked(q2)
  //      answers2_out is the interleaved list [a2_0, h2_0, a2_1, ...].
  answers2_out->clear();
  answers2_out->reserve(2 * per_col);
  for (uint64_t j = 0; j < per_col; ++j) {
    const core::Matrix& q2 = queries2[j];
    core::Matrix a2;
    core::Matrix h2;
    std::string a2_err;
    if (H1_squished.MulVecPacked(q2, 10, 3, &a2, &a2_err) != retcode::SUCCESS) {
      std::ostringstream oss;
      oss << "DoublePirProtocol::Answer: H1_squished * queries2[" << j
          << "] failed: " << a2_err;
      return fail(oss.str());
    }
    std::string h2_err;
    if (a1_t.MulVecPacked(q2, 10, 3, &h2, &h2_err) != retcode::SUCCESS) {
      std::ostringstream oss;
      oss << "DoublePirProtocol::Answer: a1_t * queries2[" << j
          << "] failed: " << h2_err;
      return fail(oss.str());
    }
    answers2_out->push_back(std::move(a2));
    answers2_out->push_back(std::move(h2));
  }
  return retcode::SUCCESS;
}

retcode DoublePirProtocol::Recover(
    uint64_t /*index*/, const core::Matrix& /*A2*/,
    const core::Matrix& /*query1*/,
    const std::vector<core::Matrix>& /*queries2*/,
    const core::Matrix& /*H2_msg*/, const core::Matrix& /*secret1*/,
    const std::vector<core::Matrix>& /*secrets2*/,
    const core::Matrix& /*answer1*/,
    const std::vector<core::Matrix>& /*answers2*/,
    const core::LweParams& /*params*/, const core::DBinfo& /*info*/,
    uint64_t* /*recovered_out*/, std::string* err) {
  if (err) {
    *err =
        "DoublePirProtocol::Recover: not yet implemented — chunk 6c "
        "of openspec task 5.5 lands the val1/val2/val3 LWE-correction "
        "chain + per-column SelectRows + Contract decode.";
  }
  return retcode::FAIL;
}

}  // namespace primihub::pir::double_pir
