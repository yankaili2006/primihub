/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir_protocol.h"

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

retcode DoublePirProtocol::Query(uint64_t /*index*/,
                                 const core::Matrix& /*A1*/,
                                 const core::Matrix& /*A2*/,
                                 const core::LweParams& /*params*/,
                                 const core::DBinfo& /*info*/,
                                 std::mt19937_64* /*noise_rng*/,
                                 core::Matrix* /*secret1_out*/,
                                 core::Matrix* /*query1_out*/,
                                 core::Matrix* /*secrets2_out*/,
                                 core::Matrix* /*queries2_out*/,
                                 std::string* err) {
  if (err) {
    *err =
        "DoublePirProtocol::Query: not yet implemented — chunk 5 of "
        "openspec task 5.5 will land the two-step LWE encrypt for "
        "the row and column indices.";
  }
  return retcode::FAIL;
}

retcode DoublePirProtocol::Answer(const core::Database& /*squished_db*/,
                                  const core::Matrix& /*H1_squished*/,
                                  const core::Matrix& /*A2_copy_transposed*/,
                                  const core::Matrix& /*query1*/,
                                  const core::Matrix& /*queries2*/,
                                  core::Matrix* /*answer1_out*/,
                                  core::Matrix* /*answer2_out*/,
                                  std::string* err) {
  if (err) {
    *err =
        "DoublePirProtocol::Answer: not yet implemented — chunk 6 of "
        "openspec task 5.5 will land squished_DB * query1 plus the "
        "per-column queries2 reduction.";
  }
  return retcode::FAIL;
}

retcode DoublePirProtocol::Recover(uint64_t /*index*/,
                                   const core::Matrix& /*query1*/,
                                   const core::Matrix& /*queries2*/,
                                   const core::Matrix& /*H2_msg*/,
                                   const core::Matrix& /*secret1*/,
                                   const core::Matrix& /*secrets2*/,
                                   const core::Matrix& /*answer1*/,
                                   const core::Matrix& /*answer2*/,
                                   const core::LweParams& /*params*/,
                                   const core::DBinfo& /*info*/,
                                   uint64_t* /*recovered_out*/,
                                   std::string* err) {
  if (err) {
    *err =
        "DoublePirProtocol::Recover: not yet implemented — chunk 6 of "
        "openspec task 5.5 will land the two-stage decode (Contract "
        "undoes the Expand from Setup, then reconstruct the cell).";
  }
  return retcode::FAIL;
}

}  // namespace primihub::pir::double_pir
