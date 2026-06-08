/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"

#include <chrono>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/double_pir/double_pir_protocol.h"
#include "src/primihub/kernel/pir/operator/double_pir/double_pir_runtime.h"
#include "src/primihub/kernel/pir/operator/pir_core/database.h"
#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

namespace {

constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kOutRecovered = "recovered";
constexpr uint64_t kRowLengthBits = 8;
constexpr uint64_t kLweSecretDim = 1024;
constexpr uint64_t kLweLogQ = 32;
// matMulVecPacked processes rows in batches of 8 — DB row count L must
// be a multiple of 8 to avoid OOB reads. (Same alignment SimplePIR
// imposes; DoublePIR shares the same kernel.)
constexpr uint64_t kRowAlignment = 8;

bool ParseU64(const std::string& s, uint64_t* out) {
  if (s.empty()) return false;
  uint64_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const uint64_t next = acc * 10 + static_cast<uint64_t>(c - '0');
    if (next < acc) return false;
    acc = next;
  }
  *out = acc;
  return true;
}

}  // namespace

retcode DoublePirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "DoublePirOperator: result is null";
    return retcode::FAIL;
  }
  if (!core::kPirCoreKernelsVendored) {
    LOG(ERROR)
        << "DoublePirOperator: pir_core kernels not vendored — build "
        << "with --define=enable_pir_core_real=1 + "
        << "--override_repository=simplepir=<path> (see openspec "
        << "task 5.5 / 7.2).";
    return retcode::FAIL;
  }

  // Parse input. Same layout as SimplePirOperator for parity:
  //   input["db_content"]    — N decimal byte strings in [0, 255]
  //   input["query_indices"] — K decimal index strings in [0, N)
  const auto db_it = input.find(kInDb);
  if (db_it == input.end() || db_it->second.empty()) {
    LOG(ERROR) << "DoublePirOperator: input missing non-empty '"
               << kInDb << "' (vector of decimal byte strings).";
    return retcode::FAIL;
  }
  const auto idx_it = input.find(kInIndices);
  if (idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "DoublePirOperator: input missing non-empty '"
               << kInIndices << "' (vector of decimal index strings).";
    return retcode::FAIL;
  }
  const auto& db_strs = db_it->second;
  const auto& idx_strs = idx_it->second;
  const uint64_t n_entries = db_strs.size();

  // Pick the DoublePIR LWE params + DB shape.
  std::string err;
  core::LweParams params;
  params.n = kLweSecretDim;
  params.logq = kLweLogQ;
  auto rc = params.Pick(/*doublepir=*/true,
                        /*samples=*/uint64_t{1} << 13, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: LweParams::Pick failed: " << err;
    return retcode::FAIL;
  }
  uint64_t l = 0, m = 0;
  rc = core::ApproxSquareDatabaseDims(n_entries, kRowLengthBits, params.p,
                                       &l, &m, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: ApproxSquareDatabaseDims failed: "
               << err;
    return retcode::FAIL;
  }
  if (l % kRowAlignment != 0) {
    LOG(ERROR) << "DoublePirOperator: derived l=" << l
               << " from N=" << n_entries
               << " is not a multiple of " << kRowAlignment
               << "; pad N so floor(sqrt(N)) is divisible by "
               << kRowAlignment << ".";
    return retcode::FAIL;
  }
  params.l = l;
  params.m = m;

  // Build DB.
  core::Database db;
  rc = db.SetupShape(n_entries, kRowLengthBits, params, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: Database::SetupShape failed: "
               << err;
    return retcode::FAIL;
  }
  // Additional DoublePIR-specific constraint: params.l must be
  // divisible by info.x (Setup precondition).
  if (params.l % db.info().x != 0) {
    LOG(ERROR) << "DoublePirOperator: params.l=" << params.l
               << " not divisible by info.x=" << db.info().x
               << "; pad N so the derived l, info.x align.";
    return retcode::FAIL;
  }
  for (uint64_t i = 0; i < n_entries; ++i) {
    uint64_t v = 0;
    if (!ParseU64(db_strs[i], &v)) {
      LOG(ERROR) << "DoublePirOperator: db_content[" << i << "]='"
                 << db_strs[i] << "' is not a decimal uint64.";
      return retcode::FAIL;
    }
    if (v > 255) {
      LOG(ERROR) << "DoublePirOperator: db_content[" << i << "]=" << v
                 << " > 255 (row_length=" << kRowLengthBits << " bits).";
      return retcode::FAIL;
    }
    const uint64_t row = i / params.m;
    const uint64_t col = i % params.m;
    db.mutable_data().Set(row, col, static_cast<uint32_t>(v));
  }
  // Shift to centered [-p/2, p/2) representation (upstream MakeDB).
  db.mutable_data().ScalarSub(static_cast<uint32_t>(params.p / 2));

  // Init + Setup once for the whole batch. Setup mutates db.info()
  // (basis/squishing) and produces H1_squished + A2_copy_transposed
  // (server state) + H2_msg (per-database public hint). We chrono-
  // instrument each stage so a single LOG(INFO) at the end can give
  // observability into where time is spent without external profiling.
  using clock = std::chrono::steady_clock;
  const auto t_init_start = clock::now();
  core::Matrix A1, A2;
  rc = double_pir::DoublePirProtocol::Init(params, db.info(), &A1, &A2, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: Init failed: " << err;
    return retcode::FAIL;
  }
  const auto t_init_end = clock::now();
  core::Matrix H1_squished, A2_copy_transposed, H2_msg;
  rc = double_pir::DoublePirProtocol::Setup(&db, A1, A2, params, &H1_squished,
                                             &A2_copy_transposed, &H2_msg,
                                             &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: Setup failed: " << err;
    return retcode::FAIL;
  }
  const auto t_setup_end = clock::now();
  const core::DBinfo info_after_setup = db.info();

  // Per-index Query / Answer / Recover loop. Noise RNG seeded once
  // per OnExecute call from std::random_device.
  std::random_device rd;
  std::mt19937_64 rng(rd());
  std::vector<std::string> recovered_strs;
  recovered_strs.reserve(idx_strs.size());
  for (const auto& idx_s : idx_strs) {
    uint64_t idx = 0;
    if (!ParseU64(idx_s, &idx)) {
      LOG(ERROR) << "DoublePirOperator: query_indices entry '"
                 << idx_s << "' is not a decimal uint64.";
      return retcode::FAIL;
    }
    if (idx >= n_entries) {
      LOG(ERROR) << "DoublePirOperator: query index " << idx
                 << " >= N=" << n_entries;
      return retcode::FAIL;
    }
    core::Matrix secret1, query1;
    std::vector<core::Matrix> secrets2, queries2;
    rc = double_pir::DoublePirProtocol::Query(idx, A1, A2, params,
                                               info_after_setup, &rng,
                                               &secret1, &query1, &secrets2,
                                               &queries2, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "DoublePirOperator: Query failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    core::Matrix answer1;
    std::vector<core::Matrix> answers2;
    rc = double_pir::DoublePirProtocol::Answer(db, H1_squished,
                                                A2_copy_transposed, query1,
                                                queries2, params, &answer1,
                                                &answers2, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "DoublePirOperator: Answer failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    uint64_t recovered = 0;
    rc = double_pir::DoublePirProtocol::Recover(idx, A2, query1, queries2,
                                                 H2_msg, secret1, secrets2,
                                                 answer1, answers2, params,
                                                 info_after_setup,
                                                 &recovered, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "DoublePirOperator: Recover failed for idx=" << idx
                 << ": " << err;
      return retcode::FAIL;
    }
    recovered_strs.push_back(std::to_string(recovered));
  }
  const auto t_end = clock::now();
  (*result)[kOutRecovered] = std::move(recovered_strs);
  using ms_d = std::chrono::duration<double, std::milli>;
  const double init_ms  = ms_d(t_init_end  - t_init_start).count();
  const double setup_ms = ms_d(t_setup_end - t_init_end).count();
  const double query_total_ms = ms_d(t_end - t_setup_end).count();
  LOG(INFO) << "DoublePirOperator: retrieved " << idx_strs.size()
            << " entries from " << n_entries << "-entry DB "
            << "(l=" << params.l << ", m=" << params.m
            << ", x=" << info_after_setup.x << ")";
  // Structured timing line — parseable as `key=value` pairs. Used by
  // bench/double_pir_latency_bench.sh in --detailed mode and useful
  // for production observability without external profilers.
  LOG(INFO) << "DoublePirOperator: timing"
            << " init_ms=" << init_ms
            << " setup_ms=" << setup_ms
            << " queries=" << idx_strs.size()
            << " query_total_ms=" << query_total_ms;
  return retcode::SUCCESS;
}

namespace {

PirCapabilities DoubleCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  // EXACTLY two servers — DoublePIR\'s privacy proof requires two
  // independently sampled secrets; using one server collapses to SimplePIR,
  // using three would need a separate three-party variant.
  caps.min_servers = 2;
  caps.max_servers = 2;
  // Per-database public hint of size O(sqrt(N) * lambda) is computed once
  // and shared with every client; clients cache it locally.
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  // The privacy guarantee REQUIRES the two servers not to collude. Selector
  // will only return this algorithm when Constraints.assume_non_colluding.
  caps.threat_model = ThreatModel::SemiHonestNonColluding;
  // Paper Table 3: ~12 ms server + ~5 ms client compute at 1 GB database.
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 1000000000ULL;  // 1e9
  caps.backends = {Backend::CPU};                   // AVX2/CUDA in P7
  // Online traffic is the algorithm\'s main win: a single LWE ciphertext
  // pair (~4 KB at 1e8) — the large cost lives in the offline hint.
  caps.typical_query_comm_bytes = 4 * 1024;
  // Pre-shared hint at 1e8 / 1 GB database is on the order of 16 MB
  // (paper Table 3). Selector uses this when the client constraint
  // includes a memory budget for the cached hint.
  caps.typical_hint_size_bytes = 16ULL * 1024 * 1024;
  // chunk 7 of openspec task 5.5 wired the full pipeline through
  // OnExecute (Init+Setup+Query+Answer+Recover); single-process
  // self-contained retrieval works. The min_servers=2 capability
  // describes the privacy threat model, not a hard runtime split.
  caps.is_real = true;
  return caps;
}

PirRegistrar<DoublePirOperator> double_pir_registrar_("double_pir",
                                                      DoubleCaps());

}  // namespace

}  // namespace primihub::pir
