/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::core::LweParams — LWE parameter struct + lookup table
 * shared by the SimplePIR / DoublePIR C++ ports. Ports upstream
 * simplepir's pir/params.go (Params struct, PickParams, Round, Delta).
 *
 * The lookup table lives in lwe_params.cc as a static
 * kLweParamEntries[] array hand-transcribed from upstream's
 * pir/params.csv at commit e9020b03 (the @simplepir pin in
 * WORKSPACE_GITHUB). The CSV is short — 9 data rows — and has not
 * changed since the upstream's USENIX'23 release, so hardcoding lets
 * this layer compile WITHOUT activating the @simplepir bazel fetch.
 *
 * Upstream invariants reflected in the table:
 *   * log_n is always 10 (LWE secret dimension N = 1024)
 *   * log_q is always 32 (ciphertext modulus q = 2^32)
 *   * sigma is always 6.4 (LWE error distribution stddev)
 *   * log_m varies 13..21 (database column count log)
 *   * p_simple (SimplePIR plaintext modulus) and p_double (DoublePIR
 *     plaintext modulus) vary per log_m
 *
 * If upstream adds rows (e.g. larger DBs, different N), the same
 * commit that bumps the @simplepir WORKSPACE_GITHUB pin must append
 * the new entries to kLweParamEntries[] in lwe_params.cc. The lookup
 * is search-by-value, not search-by-index, so old client code that
 * pinned a specific row will keep finding the same row.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_LWE_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_LWE_PARAMS_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"

namespace primihub::pir::core {

// LWE parameter bundle. Fields named to match upstream simplepir's
// Params struct for cross-reference; everything is plain uint values
// plus one double for sigma.
struct LweParams {
  uint64_t n = 0;        // LWE secret dimension (1024 in all current rows)
  double sigma = 0.0;    // LWE error stddev (6.4 in all current rows)

  uint64_t l = 0;        // DB height — set by caller before Pick()
  uint64_t m = 0;        // DB width — set by caller before Pick()

  uint64_t logq = 0;     // log2(ciphertext modulus) (32 in current rows)
  uint64_t p = 0;        // plaintext modulus, populated by Pick()

  // Delta = 2^Logq / P. Used by Round() to coarsen LWE ciphertexts to
  // plaintext slots. Fully determined by p and logq.
  uint64_t Delta() const;

  // Round(x) = ((x + Delta/2) / Delta) % p. Reduces an LWE-domain
  // value to the plaintext space.
  uint64_t Round(uint64_t x) const;

  // Lookup helper. Caller MUST set n and logq before calling. samples
  // is the maximum number of LWE samples the protocol will produce
  // (= max(L, M) for SimplePIR; passed separately for DoublePIR). If
  // doublepir is true, populates sigma and p from the p_double column;
  // otherwise from p_simple. Returns retcode::FAIL with a populated
  // err string if no matching row exists.
  retcode Pick(bool doublepir, uint64_t samples, std::string* err);
};

// One row of the embedded lookup table. Public because tests need to
// iterate it independently of LweParams::Pick().
struct LweParamEntry {
  uint64_t log_n;
  uint64_t log_m;
  uint64_t log_q;
  double sigma;
  uint64_t log_p_simple;  // not used by Pick, kept for parity / debug
  uint64_t p_simple;
  uint64_t p_double;
};

// The embedded table. Defined in lwe_params.cc.
extern const LweParamEntry kLweParamEntries[];
extern const std::size_t kLweParamEntryCount;

}  // namespace primihub::pir::core

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_LWE_PARAMS_H_
