/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// FrodoPIR (de Castro & Lee, PETS'23) — single-server LWE-based PIR with
// per-database preprocessing. Industrially deployed by Brave for the
// Privacy-Preserving STAR analytics pipeline; the maturity (vs SimplePIR
// which is a research artefact) is the main reason an operator would pick
// it.
//
// Chunk 7 (2026-06-12) flipped kIsSkeleton=false. OnExecute drives the
// full PIR pipeline end-to-end inside one process using the chunk-1..5
// C++ port of brave-experiments/frodo-pir@15573960:
//   * Decodes input["db_content"] (vector of base64-encoded fixed-size
//     element strings) via Shard::FromBase64Strings.
//   * Parses input["query_indices"] (vector of decimal index strings).
//   * For each index: fresh QueryParams::New (samples ternary secret +
//     computes lhs/rhs), GenerateQuery, Shard::Respond, parses
//     Response::ParseOutputAsBase64 into result["recovered"].
//
// Single-process model matches caps.min_servers=caps.max_servers=1
// declared in FrodoCaps(); inter-party transport via primihub
// LinkContext is N/A for FrodoPIR.
class FrodoPirOperator : public BasePirOperator {
 public:
  explicit FrodoPirOperator(const Options& options) : BasePirOperator(options) {}
  ~FrodoPirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = false;
};

}  // namespace primihub::pir
#endif
