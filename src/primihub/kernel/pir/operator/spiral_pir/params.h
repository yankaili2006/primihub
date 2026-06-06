/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_PARAMS_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/common.h"

namespace primihub::pir::spiral {

// Runtime-tunable Spiral parameters. The upstream library exposes only two
// knobs at runtime — `num_expansions` (nu_1) and `further_dims` (nu_2) —
// everything else (plaintext modulus p, q_prime_bits, gadget bases t_*,
// polynomial length) is constexpr in upstream spiral.h.
struct SpiralParams {
  uint32_t nu_1;        // num_expansions
  uint32_t nu_2;        // further_dims
  uint64_t total_n;     // 2^(nu_1 + nu_2), Spiral DB cell count
};

// Hard caps for the v1 SpiralPirOperator implementation. These reflect what
// upstream's published parameter table (all_parameter_choices.txt) covers
// for the non-stream `spiral` variant; the largest published config is the
// "wiki" entry (nu_1=9, nu_2=11, total_n=2^20=1,048,576). Records beyond
// 1M land in SpiralStream territory, a different code path we do not yet
// vendor.
inline constexpr uint64_t kMaxRecords = 1ULL << 20;       // 1,048,576

// Each Spiral cell holds at most p_bits * poly_len / 8 bytes. With the
// upstream default p=256 (8 bits) and poly_len=2048, that is 2048 bytes.
// Records above this need either record-splitting (multi-query) or the
// SpiralPack variant, neither of which v1 implements.
inline constexpr uint32_t kMaxRecordBytes = 2048;

// nu_1 / nu_2 lower bounds enforce upstream's internal assumptions. The
// upper bounds match observed values in the published table (wiki: nu_2=11,
// movie: nu_1=11). Going beyond crosses into untested territory.
inline constexpr uint32_t kMinNu = 4;
inline constexpr uint32_t kMaxNu1 = 11;
inline constexpr uint32_t kMaxNu2 = 11;

// Derive runtime Spiral params from a primihub PIR task's (num_records,
// record_size_bytes). On success returns retcode::SUCCESS and writes the
// chosen SpiralParams; on rejection returns retcode::FAIL with a populated
// error message string so the caller can surface a clear log line.
//
// Selection rule (v1):
//   total_bits = ceil(log2(num_records))
//   nu_1 = clamp(ceil(total_bits/2), kMinNu, kMaxNu1)
//   nu_2 = total_bits - nu_1
//
// This biases mildly toward smaller queries (larger nu_1 ⇒ more expansion,
// smaller compressed query) while keeping nu_2 within the published range.
// For the canonical 1M-record / 256-byte-record case the rule produces
// nu_1=10, nu_2=10 — close to the wiki entry (9, 11), within the same
// performance band (~14 KB query, ~3 s server time at 1e8 paper figures
// scaled to 1M = sub-second).
//
// Rejection modes (each returns retcode::FAIL with a distinct error string,
// suitable for log scraping and unit-test assertions):
//   * num_records < 1                        — "spiral: num_records must be >= 1"
//   * num_records > kMaxRecords              — "spiral v1 caps at 1M records ..."
//   * record_size_bytes > kMaxRecordBytes    — "spiral v1 caps at 2048 bytes per record ..."
//   * record_size_bytes < 1                  — "spiral: record_size_bytes must be >= 1"
//
// The function is pure / side-effect free and safe to call from any thread.
inline retcode EstimateParams(uint64_t num_records,
                              uint32_t record_size_bytes,
                              SpiralParams* out,
                              std::string* err) {
  if (out == nullptr || err == nullptr) {
    return retcode::FAIL;  // misuse — null out-params
  }
  if (num_records < 1) {
    *err = "spiral: num_records must be >= 1";
    return retcode::FAIL;
  }
  if (num_records > kMaxRecords) {
    *err =
        "spiral v1 caps at 1M records; for >1M use SpiralStream variant "
        "(not yet vendored — see openspec/changes/primihub-pir-multi-algo "
        "task 4.4 followup)";
    return retcode::FAIL;
  }
  if (record_size_bytes < 1) {
    *err = "spiral: record_size_bytes must be >= 1";
    return retcode::FAIL;
  }
  if (record_size_bytes > kMaxRecordBytes) {
    *err =
        "spiral v1 caps at 2048 bytes per record; larger records need "
        "SpiralPack variant or record-splitting (not yet implemented)";
    return retcode::FAIL;
  }

  // ceil(log2(num_records))
  uint32_t total_bits = 0;
  uint64_t n = num_records - 1;
  while (n > 0) {
    n >>= 1;
    ++total_bits;
  }
  if (total_bits == 0) total_bits = 1;  // 1 record → 1 bit

  // Balanced split, bias toward nu_1 for smaller queries.
  uint32_t nu_1 = (total_bits + 1) / 2;
  if (nu_1 < kMinNu) nu_1 = kMinNu;
  if (nu_1 > kMaxNu1) nu_1 = kMaxNu1;
  uint32_t nu_2 = total_bits > nu_1 ? total_bits - nu_1 : kMinNu;
  if (nu_2 < kMinNu) nu_2 = kMinNu;
  if (nu_2 > kMaxNu2) {
    // total_bits > kMaxNu1 + kMaxNu2 — would have tripped the kMaxRecords
    // gate above, so this branch is unreachable. Defensive only.
    *err = "spiral: derived params exceed published parameter table";
    return retcode::FAIL;
  }

  out->nu_1 = nu_1;
  out->nu_2 = nu_2;
  out->total_n = 1ULL << (nu_1 + nu_2);
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::spiral

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_PARAMS_H_
