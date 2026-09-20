/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_api — C++ port of upstream brave-experiments/frodo-pir
 * @15573960 src/api.rs. Provides the four user-facing FrodoPIR
 * types:
 *
 *   * Shard          — server-side bundle of Database + BaseParams.
 *                      Built from a vector of base64-encoded DB
 *                      entries; exposes Respond(query) for answering
 *                      client queries.
 *   * QueryParams    — client-side preprocessed query state: stores
 *                      lhs = b = cp.MultLeft(s) and rhs = c =
 *                      bp.MultRight(s) for a fresh random secret s.
 *                      Single-use; GenerateQuery flips `used` flag.
 *   * Query          — wraps the Vec<u32> sent over the wire to the
 *                      server. Constructed by QueryParams::Generate
 *                      Query; opaque past that.
 *   * Response       — wraps the server's Vec<u32> answer. Exposes
 *                      ParseOutputAsRow / Bytes / Base64 for the
 *                      client's post-processing.
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 5 — closes port-order #6 (api.rs, 276 LOC upstream).
 *   With chunks 1-4 covering utils + Database + BaseParams +
 *   CommonParams, this chunk is the missing protocol layer.
 *   Bundles the whole api.rs surface plus an end-to-end roundtrip
 *   test so the algorithmic loop is sealed before chunk 7 wires
 *   it into OnExecute.
 *
 * Skipped (deferred to OnExecute / chunk 7):
 *   * Shard::from_json_file / write_to_file (JSON file I/O).
 *   * bincode serialization in Respond — we return Response by
 *     value; the OnExecute layer will use primihub's own
 *     LinkContext for transport.
 *
 * Field-by-field correspondence with upstream Rust:
 *   pub struct Shard { db, base_params }                       ←→
 *       class Shard
 *   Shard::from_base64_strings(elements, dim, m, elem_size,
 *       plaintext_bits)                                        ←→
 *       Shard::FromBase64Strings(... *out, *err) -> retcode
 *   Shard::respond(q) -> ResultBoxedError<Vec<u8>>             ←→
 *       Shard::Respond(query, *out_response, *err)
 *       — out is a Response value (not pre-serialized bytes).
 *   Shard::get_db / get_base_params                            ←→
 *       GetDb / GetBaseParams const-refs.
 *
 *   pub struct QueryParams { lhs, rhs, elem_size, plaintext_bits,
 *       used }                                                 ←→
 *       class QueryParams
 *   QueryParams::new(cp, bp) -> Result<Self>                   ←→
 *       QueryParams::New(cp, bp, *out, *err) -> retcode
 *   QueryParams::generate_query(row_index)
 *       -> Result<Query, ErrorQueryParamsReused|
 *                       ErrorOverflownAdd>                     ←→
 *       QueryParams::GenerateQuery(row_index, *out_query, *err)
 *       -> retcode. Failure diagnostics reference
 *       ErrorQueryParamsReused / ErrorOverflownAdd for cross-doc
 *       traceability, matching the chunk-1 / chunk-2a convention.
 *
 *   pub struct Query(Vec<u32>)                                 ←→
 *       class Query
 *   pub struct Response(Vec<u32>)                              ←→
 *       class Response
 *   Response::parse_output_as_row / bytes / base64             ←→
 *       same names PascalCased.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_API_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_API_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"

namespace primihub::pir::frodo {

// Server's serialized PIR query — a single u32 vector of length
// `m` derived from QueryParams::lhs with `row_index`-th entry
// boosted by `get_rounding_factor(plaintext_bits)`.
class Query {
 public:
  Query() = default;
  explicit Query(std::vector<std::uint32_t> data);
  const std::vector<std::uint32_t>& AsSlice() const { return data_; }

 private:
  std::vector<std::uint32_t> data_;
};

// Server's answer to a Query — a single u32 vector of length
// `db.GetMatrixWidthSelf()` (one entry per DB-column-of-entries).
class Response {
 public:
  Response() = default;
  explicit Response(std::vector<std::uint32_t> data);

  const std::vector<std::uint32_t>& AsSlice() const { return data_; }

  // Subtracts QueryParams::rhs from each entry, divides by the
  // chunk-1 GetRoundingFactor(plaintext_bits), nearest-rounds via
  // GetRoundingFloor, mods by GetPlaintextSize. Result is a row
  // of u32 plaintext-bit chunks. Mirrors upstream parse_output_
  // as_row.
  //
  // PRECONDITION: qp.GetPlaintextBits() < 32 (else the rounding
  // factor is 0 — see chunk 1 overflow saturation). PRECONDITION:
  // qp.GetRhs().size() == this->data_.size() and >=
  // Database::GetMatrixWidth(elem_size, plaintext_bits).
  std::vector<std::uint32_t> ParseOutputAsRow(
      const class QueryParams& qp) const;

  // ParseOutputAsRow then BytesFromU32Slice. Yields the original
  // DB element as a byte vector (subject to the chunk-3c upstream
  // exact-divide quirk when elem_size % plaintext_bits == 0).
  std::vector<std::uint8_t> ParseOutputAsBytes(
      const QueryParams& qp) const;

  // ParseOutputAsRow then Base64FromU32Slice. Yields the original
  // DB element as a base64 string — the canonical way to recover
  // the upstream Database::new input.
  std::string ParseOutputAsBase64(const QueryParams& qp) const;

 private:
  std::vector<std::uint32_t> data_;
};

class QueryParams {
 public:
  QueryParams();

  // Static factory mirroring upstream QueryParams::new. Samples a
  // fresh ternary secret vector `s` of length bp.GetDim(), then
  // sets lhs = cp.MultLeft(s) and rhs = bp.MultRight(s).
  //
  // Returns retcode::FAIL on size-mismatch propagated from
  // MultLeft / MultRight.
  static retcode New(const CommonParams& cp, const BaseParams& bp,
                     QueryParams* out, std::string* err);

  // Mirrors upstream generate_query. Marks `used_` so this
  // QueryParams cannot regenerate (the LWE secret would leak on
  // reuse — same precaution as upstream ErrorQueryParamsReused).
  // Then adds GetRoundingFactor(plaintext_bits) to lhs_[row_index]
  // with overflow detection (upstream ErrorOverflownAdd).
  //
  // Failure paths:
  //   * row_index >= lhs_.size() → out-of-range
  //   * already used → "QueryParamsReused"
  //   * overflow on the boost → "OverflownAdd"
  retcode GenerateQuery(std::size_t row_index, Query* out,
                        std::string* err);

  // Accessors used by Response::ParseOutputAsRow / Bytes / Base64
  // and by tests.
  const std::vector<std::uint32_t>& GetLhs() const { return lhs_; }
  const std::vector<std::uint32_t>& GetRhs() const { return rhs_; }
  std::size_t GetElemSize() const { return elem_size_; }
  std::size_t GetPlaintextBits() const { return plaintext_bits_; }
  bool IsUsed() const { return used_; }

 private:
  std::vector<std::uint32_t> lhs_;  // b = s · A
  std::vector<std::uint32_t> rhs_;  // c = s · (A · DB)
  std::size_t elem_size_;
  std::size_t plaintext_bits_;
  bool used_;
};

class Shard {
 public:
  Shard();

  // Static factory mirroring upstream from_base64_strings.
  // Decodes the m base64 entries into a Database, then runs
  // BaseParams::NewWithSeed-equivalent via BaseParams::New
  // (fresh OS-RNG seed) to produce the precomputed RHS.
  //
  // Returns retcode::FAIL with diagnostic if Database::New or
  // BaseParams::New fails.
  static retcode FromBase64Strings(
      const std::vector<std::string>& base64_strs, std::size_t dim,
      std::size_t m, std::size_t elem_size,
      std::size_t plaintext_bits, Shard* out, std::string* err);

  // Like FromBase64Strings but takes an explicit seed for the
  // base params — used by tests + reproducible debugging. No
  // upstream counterpart (matches BaseParams::NewWithSeed pattern).
  static retcode FromBase64StringsWithSeed(
      const std::vector<std::string>& base64_strs, std::size_t dim,
      std::size_t m, std::size_t elem_size,
      std::size_t plaintext_bits, const SeedBytes& seed, Shard* out,
      std::string* err);

  // Server-side query answering. Mirrors upstream respond but
  // returns a Response value rather than bincode bytes — the
  // OnExecute layer will use primihub's own LinkContext for the
  // wire format.
  //
  // For each i in [0, db.GetMatrixWidthSelf()): response[i] =
  // db.VecMult(query.AsSlice(), i).
  //
  // PRECONDITION: query.AsSlice().size() == db.GetMatrixHeight()
  // (= m). Caller satisfies this by passing the output of
  // QueryParams::GenerateQuery on the same BaseParams.
  retcode Respond(const Query& query, Response* out,
                  std::string* err) const;

  const Database& GetDb() const { return db_; }
  const BaseParams& GetBaseParams() const { return base_params_; }

 private:
  Database db_;
  BaseParams base_params_;
};

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_API_H_
