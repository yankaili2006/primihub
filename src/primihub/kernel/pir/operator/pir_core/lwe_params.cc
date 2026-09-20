/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * LweParams implementation. Hardcoded lookup table copied from
 * upstream ahenzinger/simplepir @ e9020b03 (pir/params.csv).
 */
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"

#include <sstream>
#include <string>

#include <glog/logging.h>
#include <cmath>

namespace primihub::pir::core {

// Hand-transcribed from upstream pir/params.csv. The CSV header line
// (log(n), log(m), log(q), sigma, log(p_simple), p_simple, p_double)
// is the column order of LweParamEntry. Last verified against upstream
// commit e9020b03 (the @simplepir WORKSPACE_GITHUB pin).
const LweParamEntry kLweParamEntries[] = {
    // log_n  log_m  log_q  sigma   log_p_s  p_simple  p_double
    {  10,    13,    32,    6.4,    9,       991,      929   },
    {  10,    14,    32,    6.4,    9,       833,      781   },
    {  10,    15,    32,    6.4,    9,       701,      657   },
    {  10,    16,    32,    6.4,    9,       589,      552   },
    {  10,    17,    32,    6.4,    8,       495,      464   },
    {  10,    18,    32,    6.4,    8,       416,      390   },
    {  10,    19,    32,    6.4,    8,       350,      328   },
    {  10,    20,    32,    6.4,    8,       294,      276   },
    {  10,    21,    32,    6.4,    7,       247,      231   },
};

const std::size_t kLweParamEntryCount =
    sizeof(kLweParamEntries) / sizeof(kLweParamEntries[0]);

uint64_t LweParams::NumBasePDigits() const {
  if (p == 0) {
    LOG(FATAL) << "LweParams::NumBasePDigits called before Pick (p == 0)";
  }
  if (p == 1) {
    LOG(FATAL) << "LweParams::NumBasePDigits invalid p == 1 (log2(1) == 0)";
  }
  if (logq == 0) {
    LOG(FATAL) << "LweParams::NumBasePDigits invalid logq=" << logq;
  }
  // ceil(logq / log2(p)). Match upstream simplepir's float-based math
  // so we agree on the digit count for non-power-of-2 p (929, 781,
  // ... from the params CSV). std::log2 returns natural-rounded
  // double; the std::ceil then rounds the quotient up.
  double ratio = static_cast<double>(logq) /
                 std::log2(static_cast<double>(p));
  return static_cast<uint64_t>(std::ceil(ratio));
}

uint64_t LweParams::Delta() const {
  if (p == 0) {
    LOG(FATAL) << "LweParams::Delta called before Pick (p == 0)";
  }
  if (logq == 0 || logq > 64) {
    LOG(FATAL) << "LweParams::Delta invalid logq=" << logq;
  }
  // (1 << logq) / p — guard against logq=64 by using a wider type.
  if (logq == 64) {
    // 2^64 overflows uint64; compute via ((uint64_t{1} << 63) / p) * 2
    // exactly when p divides 2^63. Practically logq stays at 32 in
    // the current table; this branch is defensive.
    return ((static_cast<uint64_t>(1) << 63) / p) * 2;
  }
  return (static_cast<uint64_t>(1) << logq) / p;
}

uint64_t LweParams::Round(uint64_t x) const {
  const uint64_t d = Delta();
  if (d == 0) {
    LOG(FATAL) << "LweParams::Round Delta == 0 — invalid (logq, p)";
  }
  // v = (x + Delta/2) / Delta. Care: x + Delta/2 may overflow when x
  // is near uint64 max, but in practice x is bounded by 2^logq < 2^32
  // for the current table.
  const uint64_t v = (x + d / 2) / d;
  return v % p;
}

retcode LweParams::Pick(bool doublepir, uint64_t samples, std::string* err) {
  if (n == 0 || logq == 0) {
    if (err) {
      *err =
          "LweParams::Pick called with n=0 or logq=0; the caller must "
          "set both before invoking Pick.";
    }
    return retcode::FAIL;
  }

  for (std::size_t i = 0; i < kLweParamEntryCount; ++i) {
    const auto& row = kLweParamEntries[i];
    const uint64_t row_n = static_cast<uint64_t>(1) << row.log_n;
    const uint64_t row_m_cap = static_cast<uint64_t>(1) << row.log_m;
    if (n != row_n) continue;
    if (samples > row_m_cap) continue;
    if (logq != row.log_q) continue;

    sigma = row.sigma;
    p = doublepir ? row.p_double : row.p_simple;
    if (sigma == 0.0 || p == 0) {
      if (err) {
        std::ostringstream oss;
        oss << "LweParams::Pick: matching row " << i << " has invalid "
            << "sigma=" << sigma << " or p=" << p << " (doublepir="
            << (doublepir ? "true" : "false") << ")";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    return retcode::SUCCESS;
  }

  if (err) {
    std::ostringstream oss;
    oss << "LweParams::Pick: no row matches n=" << n << " logq=" << logq
        << " samples=" << samples << " doublepir="
        << (doublepir ? "true" : "false")
        << ". Available rows in the embedded table use log_n=10 "
        << "(n=1024), log_q=32 (q=2^32), log_m in [13, 21]. If you "
        << "need a row outside that range, bump the @simplepir "
        << "WORKSPACE_GITHUB pin and re-transcribe pir/params.csv "
        << "into kLweParamEntries[].";
    *err = oss.str();
  }
  return retcode::FAIL;
}

}  // namespace primihub::pir::core
