/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_params — C++ port of upstream brave-experiments/
 * frodo-pir@15573960 src/db.rs `pub struct BaseParams` and
 * `pub struct CommonParams`. These two types hold the public
 * parameters used by the FrodoPIR setup phase:
 *
 *   * BaseParams (heavy, owned by server): a 32-byte public_seed
 *     plus the precomputed RHS = A · DB matrix (one column per
 *     DB column-of-u32-entries). The server publishes these to
 *     all clients once at setup time.
 *
 *   * CommonParams (light, derivable from BaseParams):
 *     reconstructs A = generate_lwe_matrix_from_seed(public_seed,
 *     dim, m) on demand. Both client and server compute it from
 *     the published seed; it is conceptually public.
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 4 — port-order #5 follow-on to chunks 3a/3b/3c (which
 *   covered the Database container). With chunks 2a (matrices
 *   deterministic) + 2b-i/ii (SeededRng ChaCha20) + 2c (OsRng +
 *   GenerateSeed + RandomTernary) landed, the full LWE
 *   preprocessing pipeline is buildable from inside this header.
 *
 * Field-by-field correspondence with upstream Rust:
 *   pub struct BaseParams {
 *     dim, m, elem_size, plaintext_bits,
 *     public_seed: [u8; 32],
 *     rhs: Vec<Vec<u32>>,
 *   }                                                          ←→
 *       class BaseParams with the same 6 fields (rhs stored as
 *       column-form std::vector<std::vector<uint32_t>>).
 *   pub fn new(db: &Database, dim: usize) -> Self              ←→
 *       static BaseParams::New(const Database&, dim) — calls
 *       GenerateSeed for the public_seed.
 *   pub fn generate_params_rhs(db, seed, dim, m)               ←→
 *       static BaseParams::GenerateParamsRhs.
 *   pub fn mult_right(s: &[u32]) -> Result<Vec<u32>>           ←→
 *       BaseParams::MultRight(s, *out, *err) -> retcode.
 *   pub fn get_total_records / get_dim / get_elem_size /
 *       get_plaintext_bits                                     ←→
 *       BaseParams getters with the same names.
 *   pub struct CommonParams(Vec<Vec<u32>>)                     ←→
 *       class CommonParams wrapping the matrix.
 *   impl From<&BaseParams> for CommonParams                    ←→
 *       static CommonParams::FromBaseParams(BaseParams).
 *   pub fn as_matrix(&self) -> &[Vec<u32>]                     ←→
 *       CommonParams::AsMatrix() const&.
 *   pub fn mult_left(s: &[u32]) -> Result<Vec<u32>>            ←→
 *       CommonParams::MultLeft(s, *out, *err) -> retcode.
 *
 * Construction paths:
 *   * BaseParams::New(db, dim): uses GenerateSeed (OS RNG) to
 *     produce a fresh public_seed; not reproducible. Production
 *     callers use this.
 *   * BaseParams::NewWithSeed(db, dim, seed): takes an explicit
 *     seed; reproducible. Tests + debugging use this; it has no
 *     upstream counterpart but is a benign addition since the
 *     same path is reachable in upstream via mocking generate_seed.
 *
 * Skipped (deferred to OnExecute / chunk 7):
 *   * BaseParams::load / write_to_file (JSON file I/O).
 *
 * RHS shape:
 *   rhs has db.get_matrix_width_self() columns, each of length
 *   `dim`. rhs[i][j] = vec_mult(lhs[j], db column i), where
 *   lhs = SwapMatrixFmt(GenerateLweMatrixFromSeed(seed, dim, m)).
 *
 * This means MultRight(s) returns one u32 per DB-column-of-
 * entries — the server's contribution to the PIR answer.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PARAMS_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

namespace primihub::pir::frodo {

class BaseParams {
 public:
  // Default-constructible placeholder (mirrors the Database
  // pattern). Default state is empty; do not use without
  // subsequently calling New / NewWithSeed.
  BaseParams();

  // Static factory mirroring upstream Database::new but for
  // BaseParams. Generates a fresh public_seed via GenerateSeed
  // (OS RNG) — not reproducible across calls.
  //
  // Returns retcode::FAIL with *err if GenerateParamsRhs fails.
  static retcode New(const Database& db, std::size_t dim,
                     BaseParams* out, std::string* err);

  // Test-only / debugging factory taking an explicit seed.
  // Useful for reproducible tests; no upstream counterpart but
  // benign (upstream tests would mock generate_seed to achieve
  // the same effect).
  static retcode NewWithSeed(const Database& db, std::size_t dim,
                             const SeedBytes& seed, BaseParams* out,
                             std::string* err);

  // Mirrors upstream BaseParams::generate_params_rhs.
  //
  // Steps:
  //   1. lhs_seed_matrix = GenerateLweMatrixFromSeed(seed, dim, m)
  //      shape: m × dim (m columns of dim u32s each).
  //   2. lhs = SwapMatrixFmt(lhs_seed_matrix)
  //      shape: dim × m (dim columns of m u32s each).
  //   3. For each i in [0, db.GetMatrixWidthSelf()):
  //        rhs[i] = [ db.VecMult(lhs[j], i) for j in [0, dim) ]
  //   4. Return rhs (column-form, shape w × dim).
  //
  // chunk g-5: kept as a public static returning the nested form
  // so existing GenerateParamsRhs_MatchesHandComputed test +
  // any external callers compile unchanged. Internally delegates
  // to GenerateParamsRhsFlat and materialises the nested form
  // once at the boundary. Production path goes straight through
  // GenerateParamsRhsFlat via NewWithSeed -- the test-only nested
  // copy is paid only by callers that explicitly request it.
  static std::vector<std::vector<std::uint32_t>> GenerateParamsRhs(
      const Database& db, const SeedBytes& public_seed,
      std::size_t dim, std::size_t m);

  // chunk g-5: flat-buffer entry point used by NewWithSeed.
  // Mirrors GenerateParamsRhs but operates on ColMajorMatrix
  // end-to-end (no nested-vector materialisation). Uses
  // GenerateLweMatrixFromSeedFlat + SwapMatrixFmtFlat (chunk g-2 +
  // g-3) for the lhs construction, then the chunk g-4 raw-pointer
  // VecMultU32U32 overload for the dot products against db
  // columns. Result shape: height=dim, width=db.GetMatrixWidthSelf().
  static ColMajorMatrix GenerateParamsRhsFlat(
      const Database& db, const SeedBytes& public_seed,
      std::size_t dim, std::size_t m);

  // Mirrors upstream mult_right. Computes c = s · (A · DB) using
  // the precomputed RHS = A·DB. Output length = rhs.size() =
  // db.GetMatrixWidthSelf().
  //
  // Returns retcode::FAIL with *err on size mismatch between s
  // and rhs columns. On success *out is overwritten.
  retcode MultRight(const std::vector<std::uint32_t>& s,
                    std::vector<std::uint32_t>* out,
                    std::string* err) const;

  // Getters. Names match upstream (snake_case → PascalCase).
  std::size_t GetTotalRecords() const { return m_; }
  std::size_t GetDim() const { return dim_; }
  std::size_t GetElemSize() const { return elem_size_; }
  std::size_t GetPlaintextBits() const { return plaintext_bits_; }
  const SeedBytes& GetPublicSeed() const { return public_seed_; }

  // Test accessor materialising the precomputed RHS matrix into
  // the legacy nested form. Returns by value (the temporary lives
  // through the full expression that uses it; tests can also bind
  // to `const auto&` -- C++ reference lifetime extension covers
  // the by-value temporary). Production never calls this.
  std::vector<std::vector<std::uint32_t>> RhsForTest() const;

  // Direct flat-matrix accessor for callers that already speak
  // ColMajorMatrix. Skips the RhsForTest materialisation copy.
  const ColMajorMatrix& RhsFlat() const { return rhs_; }

 private:
  std::size_t dim_;
  std::size_t m_;
  std::size_t elem_size_;
  std::size_t plaintext_bits_;
  SeedBytes public_seed_;
  // chunk g-5: column-major flat storage. Shape: height = dim,
  // width = db.GetMatrixWidthSelf(). MultRight iterates columns
  // via rhs_.column_data(i) without materialising any inner
  // vector.
  ColMajorMatrix rhs_;
};

class CommonParams {
 public:
  // Default-constructible placeholder; see BaseParams for the
  // pattern rationale.
  CommonParams();

  // Construct from an already-derived matrix in the nested form.
  // Tests use this directly; FromBaseParams takes the flat-form
  // overload below.
  explicit CommonParams(std::vector<std::vector<std::uint32_t>> matrix);

  // chunk g-5: flat-form ctor for FromBaseParams. Skips the
  // nested-to-flat copy at the construction boundary.
  explicit CommonParams(ColMajorMatrix matrix);

  // Static factory mirroring upstream
  // `impl From<&BaseParams> for CommonParams`. Reconstructs A =
  // GenerateLweMatrixFromSeed(params.public_seed, params.dim,
  // params.m) — shape m × dim, column-form.
  //
  // chunk g-5: uses GenerateLweMatrixFromSeedFlat directly so the
  // production path never materialises a nested vector.
  static CommonParams FromBaseParams(const BaseParams& params);

  // Mirrors upstream as_matrix. Returns the matrix in the legacy
  // nested form (materialised on demand from the flat storage).
  // Tests can bind to `const auto&` -- C++ reference lifetime
  // extension covers the by-value temporary.
  std::vector<std::vector<std::uint32_t>> AsMatrix() const;

  // Direct flat-matrix accessor. Skips the AsMatrix materialisation
  // copy for callers that already speak ColMajorMatrix.
  const ColMajorMatrix& MatrixFlat() const { return matrix_; }

  // Mirrors upstream mult_left. Computes b[i] = vec_mult(s,
  // cols[i]) + RandomTernary(), for each i in [0, matrix_.size()).
  //
  // The RandomTernary noise makes this NON-DETERMINISTIC across
  // calls; identical inputs produce different outputs. Tests
  // check the noise distribution rather than specific values.
  //
  // Returns retcode::FAIL on size mismatch between s and matrix
  // columns.
  retcode MultLeft(const std::vector<std::uint32_t>& s,
                   std::vector<std::uint32_t>* out,
                   std::string* err) const;

 private:
  // chunk g-5: column-major flat storage for A. Shape: height = dim,
  // width = m (mirrors GenerateLweMatrixFromSeedFlat output shape).
  // MultLeft iterates columns via matrix_.column_data(i).
  ColMajorMatrix matrix_;
};

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PARAMS_H_
