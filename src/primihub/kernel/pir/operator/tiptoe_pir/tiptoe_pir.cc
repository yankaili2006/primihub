/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_pir.h"

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/registry.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "base64.h"  // NOLINT — @com_github_base64_cpp//:base64_lib

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_lhe_pir.h"
#endif

namespace primihub::pir {

namespace {
constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kOutRecovered = "recovered";
}  // namespace

#ifdef PIR_TIPTOE_RLWE_VENDORED

namespace {
// Fixed LHE/SimplePIR config for v1: secret dimension (each limb partial stays
// below the BFV plaintext modulus) + a deterministic A/secret seed.
constexpr std::uint64_t kSecretDim = 512;
constexpr std::uint64_t kSeed = 0x7E57C0DEull;

bool ParseU64(const std::string& s, std::uint64_t* out) {
  if (s.empty()) return false;
  std::uint64_t acc = 0;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
    acc = acc * 10 + static_cast<std::uint64_t>(c - '0');
  }
  *out = acc;
  return true;
}
}  // namespace

retcode TiptoePirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "TiptoePirOperator: result is null";
    return retcode::FAIL;
  }
  const auto db_it = input.find(kInDb);
  const auto idx_it = input.find(kInIndices);
  if (db_it == input.end() || db_it->second.empty() ||
      idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "TiptoePirOperator: missing non-empty '" << kInDb << "'/'"
               << kInIndices << "'";
    return retcode::FAIL;
  }
  const auto& base64_elems = db_it->second;
  const auto& idx_strs = idx_it->second;
  const std::size_t count = base64_elems.size();

  // Decode single-byte elements (v1: pt_modulus=256).
  std::vector<std::uint8_t> elems(count);
  for (std::size_t i = 0; i < count; ++i) {
    const std::string bytes = base64_decode(base64_elems[i]);
    if (bytes.size() != 1) {
      LOG(ERROR) << "TiptoePirOperator: db_content[" << i << "] is "
                 << bytes.size() << " bytes; v1 requires single-byte elements.";
      return retcode::FAIL;
    }
    elems[i] = static_cast<std::uint8_t>(bytes[0]);
  }

  // Arrange into an m x m square DB (row-major), zero-padded.
  const std::uint64_t m = static_cast<std::uint64_t>(
      std::ceil(std::sqrt(static_cast<double>(count))));
  std::vector<std::uint8_t> db(static_cast<std::size_t>(m * m), 0);
  for (std::size_t i = 0; i < count; ++i) db[i] = elems[i];

  // Build the LHE-on-SimplePIR server/client (computes the hint + H*s once).
  const tiptoe::LheSimplePir pir(db, m, kSecretDim, kSeed);

  std::vector<std::string> recovered;
  recovered.reserve(idx_strs.size());
  for (std::size_t q = 0; q < idx_strs.size(); ++q) {
    std::uint64_t idx = 0;
    if (!ParseU64(idx_strs[q], &idx) || idx >= count) {
      LOG(ERROR) << "TiptoePirOperator: query_indices[" << q << "]='"
                 << idx_strs[q] << "' invalid or out of range [0, " << count
                 << ")";
      return retcode::FAIL;
    }
    const std::uint8_t byte = pir.Retrieve(idx / m, idx % m);
    recovered.push_back(base64_encode(&byte, 1));
  }

  (*result)[kOutRecovered] = std::move(recovered);
  return retcode::SUCCESS;
}

#else  // !PIR_TIPTOE_RLWE_VENDORED

retcode TiptoePirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  (void)input;
  (void)result;
  LOG(ERROR) << "TiptoePirOperator: not yet vendored (skeleton). Build with "
                "--define=enable_tiptoe_real=1 (needs SEAL) for the real "
                "BFV-on-SimplePIR query path.";
  return retcode::FAIL;
}

#endif  // PIR_TIPTOE_RLWE_VENDORED

namespace {

PirCapabilities TiptoeCaps() {
  PirCapabilities caps;
#ifdef PIR_TIPTOE_RLWE_VENDORED
  caps.is_real = true;  // real LHE-on-SimplePIR retrieval (e2e validated)
#else
  caps.is_real = false;  // skeleton until built with enable_tiptoe_real
#endif
  caps.query_types = {QueryType::Semantic};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;   // offline per-database hint
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;
  caps.recommended_max_db_size = 100'000'000ULL;  // 1e8 (paper scale)
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 16'384;          // ~16 KB BFV query
  caps.typical_hint_size_bytes = 2'000'000'000ULL;  // ~2 GB hint at scale
  return caps;
}

PirRegistrar<TiptoePirOperator> tiptoe_registrar_("tiptoe", TiptoeCaps());

}  // namespace

}  // namespace primihub::pir
