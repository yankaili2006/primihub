/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::double_pir::DoublePirProtocol — algorithm-level static
 * functions for the DoublePIR protocol (Henzinger et al., USENIX'23).
 * Mirrors the simple_pir_protocol.{h,cc} pattern that landed in
 * commits 5a9e7616..2d77509a for SimplePIR.
 *
 * What this file (chunk 3) implements vs defers:
 *   * Init(params, info) -> samples the two public random matrices
 *     A1 (M x N) and A2 ((L / X) x N) used as shared state by every
 *     client. Pure UniformRandom — no kernel dependency. IMPLEMENTED.
 *   * Setup(db, A1, A2, params, info, ...) -> computes the public
 *     hint H2 = (H1.Expand.ConcatCols) * A2, where H1 = DB * A1.
 *     SKELETON ONLY this chunk — returns FAIL with a pointer to
 *     chunk 4 of the openspec change. Needs Database::Squish (already
 *     vendored), Matrix::Expand+ConcatCols (chunks 1/2 above), and a
 *     yet-to-add Matrix::Concat for the A2_copy row-padding step.
 *   * Query, Answer, Recover -> SKELETON ONLY this chunk; chunks 5/6
 *     will land them once Setup is end-to-end correct.
 *
 * Why ship Init alone: matches the per-chunk landing rhythm used for
 * SimplePIR (5a9e7616 lands Init+Setup together because both use the
 * same pir_core primitives; DoublePIR's Setup needs Concat which is a
 * separate primitive landing, so Setup justifies its own chunk).
 *
 * Activation: Init uses Matrix::UniformRandom which works in both
 * stub and vendored modes (no kernel call). The remaining methods
 * forward retcode::FAIL with caller-actionable `err` messages until
 * their respective chunks land.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_PROTOCOL_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_PROTOCOL_H_

#include <cstdint>
#include <random>
#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/gaussian.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::double_pir {

class DoublePirProtocol {
 public:
  // Samples the two public random matrices A1 (M x N) and A2
  // ((L / info.x) x N) used as shared state by every client. Mirrors
  // upstream simplepir/double_pir.go Init:
  //
  //   A1 := MatrixRand(p.M, p.N, p.Logq, 0)
  //   A2 := MatrixRand(p.L / info.X, p.N, p.Logq, 0)
  //
  // Pre: params must have m, n, l, and logq populated (call
  // LweParams::Pick + ApproxSquareDatabaseDims first). info.x must
  // divide params.l (LOG-FATAL otherwise — DoublePIR's compression
  // ratio requires exact partition).
  //
  // The matrices are sampled with Matrix::UniformRandom (mt19937_64 +
  // random_device). Same caveat as SimplePIR: NOT cryptographically
  // strong; production callers must inject a CSPRNG before claiming
  // LWE security. The matrices are publicly shared anyway — what
  // matters cryptographically is the secret + noise sampling in
  // Query (chunk 5), not these public matrices.
  //
  // Output: A1_out and A2_out are overwritten.
  static retcode Init(const core::LweParams& params,
                      const core::DBinfo& info,
                      core::Matrix* A1_out,
                      core::Matrix* A2_out,
                      std::string* err);

  // Computes the public per-database hint and the squished H1 the
  // server keeps online. Upstream simplepir/double_pir.go Setup:
  //
  //   H1 = DB.Data * A1
  //   H1.Transpose()
  //   H1.Expand(p.P, p.delta())
  //   H1.ConcatCols(DB.Info.X)
  //   H2 = H1 * A2
  //   DB.Data.Add(p.P / 2);  DB.Squish()
  //   H1.Add(p.P / 2);  H1.Squish(10, 3)
  //   A2_copy = A2; if A2_copy.rows % 3 != 0: append zeros to multiple of 3
  //   A2_copy.Transpose()
  //   return state=(H1_squished, A2_copy_transposed), msg=H2
  //
  // SKELETON ONLY in this chunk — returns FAIL with the "task 5.5
  // chunk 4" pointer. Lands in the next chunk once a Matrix::Concat
  // primitive is in place for the A2_copy row-padding step.
  static retcode Setup(core::Database* db,
                       const core::Matrix& A1,
                       const core::Matrix& A2,
                       const core::LweParams& params,
                       core::Matrix* H1_squished_out,
                       core::Matrix* A2_copy_transposed_out,
                       core::Matrix* H2_msg_out,
                       std::string* err);

  // SKELETON ONLY in this chunk — chunks 5/6 land Query (two-step
  // LWE encrypt for row + column indices) / Answer / Recover.
  // Signatures here are forward-declarations to let the operator
  // wiring (chunk 7 mirroring SimplePIR's 2d77509a) compile against
  // the protocol header as soon as it's stable.
  static retcode Query(uint64_t index,
                       const core::Matrix& A1,
                       const core::Matrix& A2,
                       const core::LweParams& params,
                       const core::DBinfo& info,
                       std::mt19937_64* noise_rng,
                       core::Matrix* secret1_out,
                       core::Matrix* query1_out,
                       core::Matrix* secrets2_out,
                       core::Matrix* queries2_out,
                       std::string* err);

  static retcode Answer(const core::Database& squished_db,
                        const core::Matrix& H1_squished,
                        const core::Matrix& A2_copy_transposed,
                        const core::Matrix& query1,
                        const core::Matrix& queries2,
                        core::Matrix* answer1_out,
                        core::Matrix* answer2_out,
                        std::string* err);

  static retcode Recover(uint64_t index,
                         const core::Matrix& query1,
                         const core::Matrix& queries2,
                         const core::Matrix& H2_msg,
                         const core::Matrix& secret1,
                         const core::Matrix& secrets2,
                         const core::Matrix& answer1,
                         const core::Matrix& answer2,
                         const core::LweParams& params,
                         const core::DBinfo& info,
                         uint64_t* recovered_out,
                         std::string* err);
};

}  // namespace primihub::pir::double_pir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_DOUBLE_PIR_PROTOCOL_H_
