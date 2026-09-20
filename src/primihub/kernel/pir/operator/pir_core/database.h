/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::core::Database — DBinfo metadata + Matrix-backed DB
 * for the SimplePIR / DoublePIR ports. Ports upstream simplepir's
 * pir/database.go (DBinfo struct + SetupDB + MakeRandomDB) and the
 * supporting helpers from pir/utils.go (Base_p,
 * Reconstruct_from_base_p, Compute_num_entries_base_p,
 * Num_DB_entries, ApproxSquareDatabaseDims).
 *
 * Scope at this revision — what SimplePIR.Init / Setup will call:
 *   * DBinfo struct (every field upstream defines)
 *   * NumDbEntries / ApproxSquareDatabaseDims sizing helpers
 *   * BaseP / ReconstructFromBaseP utility math
 *   * Database class with Matrix + DBinfo
 *   * MakeRandom factory (random DB for tests / benchmarks)
 *
 * Intentionally deferred (lands when DoublePIR needs them):
 *   * Squish / Unsquish (in-memory compression for the answer path)
 *   * GetElem (single-element retrieval after a query)
 *   * MakeDB (build from user-supplied vals — Base_p packing)
 *   * Database<->wire serialization (lives in a separate IO layer)
 *
 * Like Matrix, this layer's kernel-touching code paths bifurcate on
 * PIR_PIR_CORE_REAL via the BUILD select(). MakeRandom uses Matrix's
 * pure-arithmetic UniformRandom path, so it works in BOTH modes —
 * vendored mode adds nothing extra here. Future SimplePIR Setup will
 * call Matrix::Mul to compute the per-database hint, which is when
 * vendoring becomes mandatory.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_DATABASE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_DATABASE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

namespace primihub::pir::core {

// Mirrors upstream simplepir's DBinfo. Every field has the same name
// as the Go version (snake-cased) so the future port can be diffed
// against pir/database.go line by line.
struct DBinfo {
  uint64_t num = 0;          // number of DB entries
  uint64_t row_length = 0;   // bits per DB entry

  // If log(p) > row_length, multiple DB entries pack into one Z_p
  // element; `packing` is the count of entries per element.
  // Otherwise (entry > log(p)), `ne` is the count of Z_p elements
  // needed to represent one entry.
  uint64_t packing = 0;
  uint64_t ne = 0;

  // Tunable repetition parameter. Must be in [1, ne] and divide ne.
  // For the first SimplePIR port pass this is always set equal to ne.
  uint64_t x = 0;

  uint64_t p = 0;            // plaintext modulus (mirrors LweParams.p)
  uint64_t logq = 0;         // log2(ciphertext modulus)

  // In-memory compression scratch (Squish / Unsquish). Set to 0 in
  // this revision; lands when DoublePIR needs them.
  uint64_t basis = 0;
  uint64_t squishing = 0;
  uint64_t cols = 0;
};

// --- Utility math ---------------------------------------------------

// Returns the i-th elem in the base-p decomposition of m.
// E.g., BaseP(10, 12345, 0) == 5, BaseP(10, 12345, 1) == 4.
uint64_t BaseP(uint64_t p, uint64_t m, uint64_t i);

// Inverse of BaseP. ReconstructFromBaseP(p, [v0, v1, v2, ...]) =
// v0 + v1*p + v2*p^2 + ...
uint64_t ReconstructFromBaseP(uint64_t p, const uint64_t* vals,
                              std::size_t n);

// Returns ceil(log_q / log_p). The number of Z_p elements needed to
// represent a single Z_q element.
uint64_t ComputeNumEntriesBaseP(uint64_t p, uint64_t log_q);

// Sizes for a database of N entries each `row_length` bits when stored
// in Z_p elements. Outputs:
//   db_elems       — total number of Z_p elems to allocate (l * m >= this)
//   elems_per_entry — DBinfo.ne (1 when packing is in use)
//   entries_per_elem — DBinfo.packing (0 when not packing)
// Returns retcode::FAIL if the math degenerates (e.g., N == 0).
retcode NumDbEntries(uint64_t n, uint64_t row_length, uint64_t p,
                     uint64_t* db_elems, uint64_t* elems_per_entry,
                     uint64_t* entries_per_elem, std::string* err);

// Picks the smallest (l, m) such that l*m >= db_elems and ne divides
// l. Output is l ~ sqrt(db_elems) rounded up to the next multiple of
// ne. Returns retcode::FAIL on degenerate input.
retcode ApproxSquareDatabaseDims(uint64_t n, uint64_t row_length,
                                  uint64_t p,
                                  uint64_t* l, uint64_t* m,
                                  std::string* err);

// Reconstruct a DB entry from base-p decomposed Z_p values. Mirrors
// upstream simplepir's ReconstructElem in pir/database.go.
//
// Inputs:
//   * `vals` — info.ne base-p digits that the Recover loop produced
//     after applying Round() to the noisy answer cells. Passed by
//     value because upstream mutates it (adds info.p/2 and reduces
//     by p) before recombining; passing by value spares the caller.
//   * `index` — the DB entry index the client originally queried. Only
//     used when info.packing > 0 to extract the right sub-entry.
//   * `info` — must have p, logq, packing, ne, row_length populated.
//
// Output: the recovered Z_p element (or, when info.packing > 0, the
// recovered row_length-bit sub-entry of the packed element).
uint64_t ReconstructElem(std::vector<uint64_t> vals, uint64_t index,
                          const DBinfo& info);

// --- Database class -------------------------------------------------

class Database {
 public:
  Database() = default;

  const DBinfo& info() const { return info_; }
  const Matrix& data() const { return data_; }
  Matrix& mutable_data() { return data_; }

  // Allocates info_ + a zero-filled Matrix sized for `params`. Caller
  // must populate data_ either directly (Set/MatrixAdd) or via
  // MakeRandom. Used by the SimplePIR Setup path.
  retcode SetupShape(uint64_t num, uint64_t row_length,
                     const LweParams& params, std::string* err);

  // Allocates info_ + fills data_ with uniform random Z_p values, then
  // shifts to [-p/2, p/2] via ScalarSub(p/2). Equivalent of upstream's
  // MakeRandomDB. Useful for unit tests + correctness benchmarks
  // where the DB content does not matter, only the protocol latency
  // and noise behavior.
  static Database MakeRandom(uint64_t num, uint64_t row_length,
                             const LweParams& params, std::string* err,
                             retcode* rc);

  // Direct mutable access to the DBinfo struct. Most callers should
  // not need to mutate it directly — Squish/Unsquish update the
  // relevant fields themselves. Exposed for tests + the protocol
  // layer that reads fields like `ne` after SetupShape.
  DBinfo& mutable_info() { return info_; }

  // Squish — in-memory compression mirroring upstream's
  //   DB.Info.Basis = 10;  DB.Info.Squishing = 3;  DB.Info.Cols = data.cols
  //   DB.Data.Squish(basis, squishing)
  // Returns FAIL if the params do not allow this compression
  // (p > 2^basis OR logq < basis * squishing) — upstream panics on
  // the same condition; we surface a FAIL with `err`. Caller must
  // pass a populated LweParams whose `p` and `logq` match the
  // database; we read them off info_ which Setup populated.
  //
  // POST: info_.basis = basis, info_.squishing = squishing,
  // info_.cols = old_cols, data_ has ceil(old_cols / squishing)
  // columns.
  retcode Squish(uint64_t basis, uint64_t squishing, std::string* err);

  // Unsquish — inverse of Squish. Uses info_.basis / info_.squishing /
  // info_.cols stashed during Squish, so callers do not have to
  // re-pass them. Returns FAIL if the Database was never Squished
  // (info_.squishing == 0). POST: data_ restored to (l x cols), info_
  // squishing/basis/cols zeroed.
  retcode Unsquish(std::string* err);

 private:
  DBinfo info_;
  Matrix data_;
};

}  // namespace primihub::pir::core

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_DATABASE_H_
