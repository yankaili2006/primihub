/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir.h"

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_protocol.h"

namespace primihub::pir {

namespace {

constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kOutRecovered = "recovered";
constexpr uint64_t kRowLengthBits = 8;
constexpr uint64_t kLweSecretDim = 1024;
constexpr uint64_t kLweLogQ = 32;
constexpr uint64_t kSquishBasis = 10;
constexpr uint64_t kSquishingFactor = 3;
// matMulVecPacked processes rows in batches of 8; the DB row count L
// MUST be a multiple of 8 to avoid OOB reads.
constexpr uint64_t kRowAlignment = 8;

// Parse a decimal string into a uint64. Returns false on parse error,
// without throwing — std::stoull would throw on garbage which is
// awkward to propagate through retcode.
bool ParseU64(const std::string& s, uint64_t* out) {
  if (s.empty()) return false;
  uint64_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const uint64_t next = acc * 10 + static_cast<uint64_t>(c - '0');
    if (next < acc) return false;  // overflow
    acc = next;
  }
  *out = acc;
  return true;
}

}  // namespace

retcode SimplePirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "SimplePirOperator: result is null";
    return retcode::FAIL;
  }
  if (!core::kPirCoreKernelsVendored) {
    LOG(ERROR)
        << "SimplePirOperator: pir_core kernels not vendored — build "
        << "with --define=enable_pir_core_real=1 and provide @simplepir "
        << "bazel override (see openspec task 7.2 Phase 6).";
    return retcode::FAIL;
  }
  // Parse input. Expected layout:
  //   input["db_content"]    — N decimal byte strings in [0, 255]
  //   input["query_indices"] — K decimal index strings in [0, N)
  const auto db_it = input.find(kInDb);
  if (db_it == input.end() || db_it->second.empty()) {
    LOG(ERROR) << "SimplePirOperator: input missing non-empty '"
               << kInDb << "' (vector of decimal byte strings).";
    return retcode::FAIL;
  }
  const auto idx_it = input.find(kInIndices);
  if (idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "SimplePirOperator: input missing non-empty '"
               << kInIndices << "' (vector of decimal index strings).";
    return retcode::FAIL;
  }
  const auto& db_strs = db_it->second;
  const auto& idx_strs = idx_it->second;
  const uint64_t n_entries = db_strs.size();

  // Pick LWE params + DB shape.
  std::string err;
  core::LweParams params;
  params.n = kLweSecretDim;
  params.logq = kLweLogQ;
  // samples=2^13 fits log_m=13 row of the embedded table; bumps up
  // automatically as m grows because Pick scans entries in row order
  // looking for the smallest log_m row that fits.
  auto rc = params.Pick(/*doublepir=*/false,
                        /*samples=*/uint64_t{1} << 13, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: LweParams::Pick failed: " << err;
    return retcode::FAIL;
  }
  uint64_t l = 0, m = 0;
  rc = core::ApproxSquareDatabaseDims(n_entries, kRowLengthBits, params.p,
                                       &l, &m, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: ApproxSquareDatabaseDims failed: "
               << err;
    return retcode::FAIL;
  }
  if (l % kRowAlignment != 0) {
    LOG(ERROR) << "SimplePirOperator: derived l=" << l << " from N="
               << n_entries
               << " is not a multiple of " << kRowAlignment
               << "; matMulVecPacked kernel processes rows in batches of "
               << kRowAlignment
               << " and reads OOB otherwise. Pad N so that"
               << " floor(sqrt(N)) is divisible by " << kRowAlignment
               << " (e.g. N=64 -> l=8, N=576 -> l=24).";
    return retcode::FAIL;
  }
  params.l = l;
  params.m = m;

  // Build DB.
  core::Database db;
  rc = db.SetupShape(n_entries, kRowLengthBits, params, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: Database::SetupShape failed: "
               << err;
    return retcode::FAIL;
  }
  for (uint64_t i = 0; i < n_entries; ++i) {
    uint64_t v = 0;
    if (!ParseU64(db_strs[i], &v)) {
      LOG(ERROR) << "SimplePirOperator: db_content[" << i
                 << "]='" << db_strs[i]
                 << "' is not a decimal uint64.";
      return retcode::FAIL;
    }
    if (v > 255) {
      LOG(ERROR) << "SimplePirOperator: db_content[" << i << "]=" << v
                 << " > 255 (row_length=" << kRowLengthBits << " bits).";
      return retcode::FAIL;
    }
    const uint64_t row = i / params.m;
    const uint64_t col = i % params.m;
    db.mutable_data().Set(row, col, static_cast<uint32_t>(v));
  }
  // Shift to the centered representation [-p/2, p/2) — matches
  // upstream simplepir's MakeDB convention. Without this,
  // ReconstructElem's add-p/2 step recovers (byte + p/2) mod p
  // instead of the original byte (see simple_pir_protocol_test
  // EndToEndRetrievesCorrectEntry comments for the derivation).
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));

  // Run Init + Setup + Squish + GenSecret once for the whole batch.
  core::Matrix A;
  rc = simple_pir::SimplePirProtocol::Init(params, &A, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: Init failed: " << err;
    return retcode::FAIL;
  }
  core::Matrix H;
  rc = simple_pir::SimplePirProtocol::Setup(&db, A, params, &H, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: Setup failed: " << err;
    return retcode::FAIL;
  }
  rc = db.Squish(kSquishBasis, kSquishingFactor, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: Database::Squish failed: " << err;
    return retcode::FAIL;
  }
  core::Matrix secret;
  rc = simple_pir::SimplePirProtocol::GenSecret(params, &secret, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: GenSecret failed: " << err;
    return retcode::FAIL;
  }

  // Per-index Query/Answer/Recover loop. Noise RNG seeded fresh from
  // std::random_device per OnExecute call; production callers that need
  // a CSPRNG should wire one in via a future Options field.
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::vector<std::string> recovered_strs;
  recovered_strs.reserve(idx_strs.size());
  for (const auto& idx_s : idx_strs) {
    uint64_t idx = 0;
    if (!ParseU64(idx_s, &idx)) {
      LOG(ERROR) << "SimplePirOperator: query_indices entry '"
                 << idx_s << "' is not a decimal uint64.";
      return retcode::FAIL;
    }
    if (idx >= n_entries) {
      LOG(ERROR) << "SimplePirOperator: query index " << idx
                 << " >= N=" << n_entries;
      return retcode::FAIL;
    }
    core::Matrix query;
    rc = simple_pir::SimplePirProtocol::Query(
        idx, A, secret, params, db.info(), &rng, &query, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "SimplePirOperator: Query failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    core::Matrix answer;
    rc = simple_pir::SimplePirProtocol::Answer(db, query, &answer, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "SimplePirOperator: Answer failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    uint64_t recovered = 0;
    rc = simple_pir::SimplePirProtocol::Recover(
        idx, query, H, secret, answer, params, db.info(), &recovered, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "SimplePirOperator: Recover failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    recovered_strs.push_back(std::to_string(recovered));
  }
  (*result)[kOutRecovered] = std::move(recovered_strs);
  LOG(INFO) << "SimplePirOperator: retrieved " << idx_strs.size()
            << " entries from " << n_entries << "-entry DB "
            << "(l=" << params.l << ", m=" << params.m << ")";
  return retcode::SUCCESS;
}

namespace {

PirCapabilities SimpleCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;       // public matrix A is pre-shared
  caps.hint_per_database = true;      // H = A * D depends on D
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;  // ~300 ms at 1 GB per paper §6
  caps.recommended_max_db_size = 100'000'000ULL;  // 1e8 sweet spot
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 121 * 1024;  // ~121 KB (paper Table 2)
  caps.typical_hint_size_bytes = 121ULL * 1024 * 1024;  // ~121 MB at 1 GB
  return caps;
}

PirRegistrar<SimplePirOperator> simple_pir_registrar_("simple_pir", SimpleCaps());

}  // namespace

}  // namespace primihub::pir
