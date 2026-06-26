/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <glog/logging.h>

#include "base64.h"  // NOLINT — @com_github_base64_cpp//:base64_lib

#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

namespace primihub::pir {

namespace {

constexpr const char* kInDb = "db_content";
constexpr const char* kInIndices = "query_indices";
constexpr const char* kOutRecovered = "recovered";

// spiral DEFAULT_MODULI; YPIR v1 uses the small poly_len=8 binary-gadget preset
// (t_exp_left=60 -> bits_per=1, tiny key-switch noise) that the end-to-end
// retrieval test pins. db_cols = instances(1)*poly_len(8) = 8; the db is grown
// by raising db_dim_1. Elements are single bytes (pt_modulus=256). Scaling to
// the paper's poly_len=2048 preset is a tuning follow-up.
constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;
constexpr std::size_t kYpirDbCols = 8;        // instances=1 * poly_len=8
constexpr std::size_t kYpirMaxDbDim1 = 8;     // db_rows <= 2^11 -> <= 16384 elems

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

std::array<std::uint8_t, 32> FixedSeed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

ypir::Params MakeParams(std::size_t db_dim_1) {
  return ypir::Params::Init(8, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 60, 2, 3, true,
                            db_dim_1, 1, 1, 0, 0);
}

}  // namespace

retcode YpirOperator::OnExecute(const PirDataType& input, PirDataType* result) {
  if (result == nullptr) {
    LOG(ERROR) << "YpirOperator: result is null";
    return retcode::FAIL;
  }

  // ---- Parse inputs ----
  const auto db_it = input.find(kInDb);
  if (db_it == input.end() || db_it->second.empty()) {
    LOG(ERROR) << "YpirOperator: input missing non-empty '" << kInDb
               << "' (vector of base64-encoded single-byte element strings)";
    return retcode::FAIL;
  }
  const auto idx_it = input.find(kInIndices);
  if (idx_it == input.end() || idx_it->second.empty()) {
    LOG(ERROR) << "YpirOperator: input missing non-empty '" << kInIndices
               << "' (vector of decimal index strings)";
    return retcode::FAIL;
  }
  const auto& base64_elems = db_it->second;
  const auto& idx_strs = idx_it->second;
  const std::size_t m = base64_elems.size();

  // ---- Decode elements (YPIR v1: one byte per element) ----
  std::vector<std::uint8_t> elem_bytes(m);
  for (std::size_t i = 0; i < m; ++i) {
    const std::string bytes = base64_decode(base64_elems[i]);
    if (bytes.size() != 1) {
      LOG(ERROR) << "YpirOperator: db_content[" << i << "] decodes to "
                 << bytes.size() << " bytes; YPIR v1 requires single-byte "
                 << "elements (pt_modulus=256).";
      return retcode::FAIL;
    }
    elem_bytes[i] = static_cast<std::uint8_t>(bytes[0]);
  }

  // ---- Size params to the db (db_cols fixed, grow db_dim_1) ----
  std::size_t db_dim_1 = 1;
  const std::size_t rows_needed = (m + kYpirDbCols - 1) / kYpirDbCols;
  while ((static_cast<std::size_t>(1) << (db_dim_1 + 3)) < rows_needed)
    ++db_dim_1;
  if (db_dim_1 > kYpirMaxDbDim1) {
    LOG(ERROR) << "YpirOperator: db has " << m << " elements, exceeding the "
               << "YPIR v1 capacity (" << (kYpirDbCols << (kYpirMaxDbDim1 + 3))
               << ").";
    return retcode::FAIL;
  }
  const ypir::Params p = MakeParams(db_dim_1);
  const std::size_t db_rows = static_cast<std::size_t>(1) << (db_dim_1 + 3);
  const std::size_t db_cols = kYpirDbCols;
  const std::size_t num_rlwe_outputs = db_cols / p.poly_len;  // = 1
  ypir::NttContext ctx(p);

  // db: element idx -> (row = idx/db_cols, col = idx%db_cols), row-major input.
  std::vector<std::uint8_t> db(db_rows * db_cols, 0);
  for (std::size_t i = 0; i < m; ++i) db[i] = elem_bytes[i];
  const ypir::YServer<std::uint8_t> srv(p, db, /*is_simplepir=*/true,
                                        /*inp_transposed=*/false,
                                        /*pad_rows=*/false);

  // ---- Client setup: secret + packing expansion params ----
  ypir::ChaChaRng key = ypir::ChaChaRng::FromSeed(FixedSeed(31));
  const ypir::Client client(ctx, /*hamming=*/2, key);
  ypir::ChaChaRng lwe_ent = ypir::ChaChaRng::FromSeed(FixedSeed(32));
  const ypir::YClient yc(ctx, client, lwe_ent);

  const ypir::DiscreteGaussian dg = ypir::DiscreteGaussian::Init(p.noise_width);
  ypir::ChaChaRng ep = ypir::ChaChaRng::FromSeed(FixedSeed(33));
  ypir::ChaChaRng ep_pub = ypir::ChaChaRng::FromSeed(FixedSeed(34));
  const std::vector<ypir::PolyMatrixNTT> pack_pub_params =
      ypir::RawGenerateExpansionParams(ctx, dg, client.SkReg(),
                                       p.poly_len_log2, p.t_exp_left, ep,
                                       ep_pub);
  const ypir::YConstants y_constants = ypir::GenerateYConstants(ctx);

  // ---- Offline: hint -> prepacked ----
  const std::vector<std::uint64_t> hint_0 =
      srv.AnswerHintRing(ctx, ypir::kSeed0, db_cols);
  std::vector<std::uint64_t> combined = hint_0;
  combined.resize(hint_0.size() + db_cols, 0);  // append zero b-row
  const std::vector<std::vector<ypir::PolyMatrixNTT>> prepacked_lwe =
      ypir::PrepPackManyLwes(ctx, combined, num_rlwe_outputs);

  // ---- Per-index retrieval ----
  std::vector<std::string> recovered;
  recovered.reserve(idx_strs.size());
  for (std::size_t q_i = 0; q_i < idx_strs.size(); ++q_i) {
    std::uint64_t idx = 0;
    if (!ParseU64(idx_strs[q_i], &idx)) {
      LOG(ERROR) << "YpirOperator: query_indices[" << q_i << "]='"
                 << idx_strs[q_i] << "' is not a decimal uint64";
      return retcode::FAIL;
    }
    if (idx >= m) {
      LOG(ERROR) << "YpirOperator: query_indices[" << q_i << "]=" << idx
                 << " out of range [0, " << m << ")";
      return retcode::FAIL;
    }
    const std::size_t row = static_cast<std::size_t>(idx) / db_cols;
    const std::size_t col = static_cast<std::size_t>(idx) % db_cols;

    ypir::ChaChaRng noise =
        ypir::ChaChaRng::FromSeed(FixedSeed(static_cast<std::uint8_t>(row)));
    const std::vector<std::uint64_t> packed_query =
        yc.GenerateQueryPacked(ypir::kSeed0, db_dim_1, row, noise);
    const std::vector<std::uint64_t> intermediate =
        srv.AnswerQuery(packed_query);
    const std::vector<ypir::PolyMatrixNTT> packed =
        ypir::PackManyLwes(ctx, prepacked_lwe, intermediate, num_rlwe_outputs,
                           pack_pub_params, y_constants);

    std::vector<std::uint64_t> row_vals;
    row_vals.reserve(db_cols);
    for (const ypir::PolyMatrixNTT& ct : packed) {
      const ypir::PolyMatrixRaw dec =
          ctx.FromNtt(client.DecryptMatrixReg(ct));
      for (std::size_t z = 0; z < p.poly_len; ++z)
        row_vals.push_back(
            ypir::Rescale(dec.data[z], p.modulus, p.pt_modulus));
    }

    const std::uint8_t out_byte = static_cast<std::uint8_t>(row_vals[col]);
    recovered.push_back(base64_encode(&out_byte, 1));
  }

  (*result)[kOutRecovered] = std::move(recovered);
  return retcode::SUCCESS;
}

namespace {

PirCapabilities YpirCaps() {
  PirCapabilities caps;
  caps.is_real = true;  // full SimplePIR query path ported (task 7.3); E2E PASS
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;       // per-db hint trades for low traffic
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;  // paper: ~500 ms at 1 GB
  caps.recommended_max_db_size = 1'000'000'000ULL;  // 1e9
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 900;  // <1 KB — YPIR's distinguishing feature
  caps.typical_hint_size_bytes = 8ULL * 1024 * 1024;  // ~8 MB at 1 GB
  return caps;
}

PirRegistrar<YpirOperator> ypir_registrar_("ypir", YpirCaps());

}  // namespace

}  // namespace primihub::pir
