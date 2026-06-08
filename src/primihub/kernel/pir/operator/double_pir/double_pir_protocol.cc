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

retcode DoublePirProtocol::Setup(core::Database* /*db*/,
                                 const core::Matrix& /*A1*/,
                                 const core::Matrix& /*A2*/,
                                 const core::LweParams& /*params*/,
                                 core::Matrix* /*H1_squished_out*/,
                                 core::Matrix* /*A2_copy_transposed_out*/,
                                 core::Matrix* /*H2_msg_out*/,
                                 std::string* err) {
  if (err) {
    *err =
        "DoublePirProtocol::Setup: not yet implemented — chunk 4 of "
        "openspec/changes/primihub-pir-multi-algo task 5.5 will land "
        "DB*A1, Transpose, Expand, ConcatCols, H1*A2, Squish, and "
        "A2_copy row-padding (needs a Matrix::Concat primitive that "
        "is not yet vendored).";
  }
  return retcode::FAIL;
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
