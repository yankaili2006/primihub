/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_pir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "base64.h"  // NOLINT — @com_github_base64_cpp//:base64_lib

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_api.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

namespace {

constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kOutRecovered = "recovered";

// Default protocol parameters. Match an upstream test fixture:
//   m = db.size() (from input)
//   elem_size = decoded-byte-length-of-each-DB-entry × 8 (must be uniform)
//   plaintext_bits = 10  (upstream test client_query_to_server_attempt_
//                         params_reuse uses this; small enough to avoid
//                         the chunk-1 plaintext_bits>=32 saturation)
//   dim = 512            (upstream's lwe_dim default in both tests)
// Production deployments can plug different values via a future Options
// extension; for chunk-7 single-process operation these defaults work.
constexpr std::size_t kFrodoLweDim = 512;
constexpr std::size_t kFrodoPlaintextBits = 10;

bool ParseU64(const std::string& s, std::uint64_t* out) {
  if (s.empty()) return false;
  std::uint64_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    const std::uint64_t next = acc * 10 + static_cast<std::uint64_t>(c - '0');
    if (next < acc) return false;  // overflow
    acc = next;
  }
  *out = acc;
  return true;
}

}  // namespace

retcode FrodoPirOperator::OnExecute(const PirDataType& input,
                                    PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "FrodoPirOperator: result is null";
    return retcode::FAIL;
  }
  // ---- Parse inputs ----
  const auto db_it = input.find(kInDb);
  if (db_it == input.end() || db_it->second.empty()) {
    LOG(ERROR) << "FrodoPirOperator: input missing non-empty '"
               << kInDb << "' (vector of base64-encoded element strings)";
    return retcode::FAIL;
  }
  const auto idx_it = input.find(kInIndices);
  if (idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "FrodoPirOperator: input missing non-empty '"
               << kInIndices << "' (vector of decimal index strings)";
    return retcode::FAIL;
  }
  const auto& base64_elems = db_it->second;
  const auto& idx_strs = idx_it->second;
  const std::size_t m = base64_elems.size();

  // ---- Determine elem_size from first entry; require uniform size ----
  // FrodoPIR Database stores rows of equal length; mixed-size inputs
  // would silently truncate. Reject explicitly.
  const std::string first_bytes = base64_decode(base64_elems[0]);
  if (first_bytes.empty()) {
    LOG(ERROR) << "FrodoPirOperator: db_content[0] base64 decoded to "
               << "empty bytes — input is empty or not base64";
    return retcode::FAIL;
  }
  const std::size_t elem_size_bytes = first_bytes.size();
  for (std::size_t i = 1; i < m; ++i) {
    const std::string b = base64_decode(base64_elems[i]);
    if (b.size() != elem_size_bytes) {
      LOG(ERROR) << "FrodoPirOperator: db_content[" << i << "] decodes "
                 << "to " << b.size() << " bytes; db_content[0] decoded "
                 << "to " << elem_size_bytes
                 << " bytes. FrodoPIR requires uniform-size DB entries.";
      return retcode::FAIL;
    }
  }
  const std::size_t elem_size = elem_size_bytes * 8;

  // ---- Build the Shard ----
  frodo::Shard shard;
  std::string err;
  retcode rc = frodo::Shard::FromBase64Strings(base64_elems, kFrodoLweDim, m,
                                         elem_size, kFrodoPlaintextBits,
                                         &shard, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "FrodoPirOperator: Shard::FromBase64Strings failed: "
               << err;
    return retcode::FAIL;
  }
  const auto& bp = shard.GetBaseParams();
  const auto cp = frodo::CommonParams::FromBaseParams(bp);

  // ---- Per-index PIR retrieval loop ----
  std::vector<std::string> recovered;
  recovered.reserve(idx_strs.size());
  for (std::size_t q_i = 0; q_i < idx_strs.size(); ++q_i) {
    std::uint64_t idx = 0;
    if (!ParseU64(idx_strs[q_i], &idx)) {
      LOG(ERROR) << "FrodoPirOperator: query_indices[" << q_i
                 << "]='" << idx_strs[q_i]
                 << "' is not a decimal uint64";
      return retcode::FAIL;
    }
    if (idx >= m) {
      LOG(ERROR) << "FrodoPirOperator: query_indices[" << q_i
                 << "]=" << idx << " out of range [0, " << m << ")";
      return retcode::FAIL;
    }
    // Fresh QueryParams per retrieval — single-use enforcement
    // prevents LWE secret leakage from re-using the same s.
    frodo::QueryParams qp;
    rc = frodo::QueryParams::New(cp, bp, &qp, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "FrodoPirOperator: QueryParams::New failed at "
                 << "query " << q_i << ": " << err;
      return retcode::FAIL;
    }
    frodo::Query q;
    rc = qp.GenerateQuery(static_cast<std::size_t>(idx), &q, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "FrodoPirOperator: GenerateQuery failed at query "
                 << q_i << " (idx=" << idx << "): " << err;
      return retcode::FAIL;
    }
    frodo::Response resp;
    rc = shard.Respond(q, &resp, &err);
    if (rc != retcode::SUCCESS) {
      LOG(ERROR) << "FrodoPirOperator: Shard::Respond failed at query "
                 << q_i << ": " << err;
      return retcode::FAIL;
    }
    recovered.push_back(resp.ParseOutputAsBase64(qp));
  }

  (*result)[kOutRecovered] = std::move(recovered);
  return retcode::SUCCESS;
}

namespace {

PirCapabilities FrodoCaps() {
  PirCapabilities caps;
  caps.is_real = true;  // chunk 7 wired full pipeline through OnExecute
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  // Paper §5: ~30 ms server-side at 1 GB; classify as Ms.
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 100'000'000ULL;
  caps.backends = {Backend::CPU};
  // FrodoPIR's online traffic is smaller than SimplePIR (paper Table 1).
  caps.typical_query_comm_bytes = 64 * 1024;
  caps.typical_hint_size_bytes = 40ULL * 1024 * 1024;  // ~40 MB at 1 GB
  return caps;
}

PirRegistrar<FrodoPirOperator> frodo_pir_registrar_("frodo_pir", FrodoCaps());

}  // namespace

}  // namespace primihub::pir
