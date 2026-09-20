/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_pir_latency_bench — drives FrodoPIR at caller-supplied scale
 * and prints per-stage wall-clock latency to stdout. Times Shard
 * construction (Setup), per-query QueryParams::New + GenerateQuery +
 * Respond + Parse separately so the CSV captures the same shape as
 * double_pir_latency_bench / simple_pir_latency_bench.
 *
 * Usage:
 *   frodo_pir_latency_bench --n N --queries Q [--csv]
 *
 *   --n N         number of byte entries in the synthetic DB. Any
 *                 positive integer works (FrodoPIR has no row-
 *                 alignment constraint like DoublePIR).
 *   --queries Q   number of queries to issue (used for the per-query
 *                 latency average). Q indices are picked deterministically
 *                 across [0, N).
 *   --csv         emit a single CSV line "n,setup_ms,per_query_ms,
 *                 wall_ms,status" (default: a human-readable block).
 *
 * Drives the algorithm directly through Shard / QueryParams / Response
 * rather than FrodoPirOperator::OnExecute. The operator is single-process
 * and exercises the same code path; bypassing it gives clean per-stage
 * timing without needing a timing LOG inside the operator.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "base64.h"  // NOLINT

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_api.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"

namespace {

double ms_since(const std::chrono::steady_clock::time_point& start) {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t n = 1024;
  uint64_t queries = 16;
  bool csv = false;
  // Defaults match FrodoPirOperator::OnExecute (chunk 7).
  std::size_t lwe_dim = 512;
  std::size_t plaintext_bits = 10;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n" && i + 1 < argc) {
      n = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--queries" && i + 1 < argc) {
      queries = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--lwe-dim" && i + 1 < argc) {
      lwe_dim = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--plaintext-bits" && i + 1 < argc) {
      plaintext_bits = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--csv") {
      csv = true;
    } else if (a == "-h" || a == "--help") {
      std::fprintf(stderr,
                   "Usage: %s --n N --queries Q [--csv] "
                   "[--lwe-dim D=512] [--plaintext-bits B=10]\n",
                   argv[0]);
      return 0;
    }
  }
  if (n == 0 || queries == 0) {
    std::fprintf(stderr, "n and queries must be > 0\n");
    return 2;
  }

  // Build a deterministic single-byte DB.
  std::vector<std::string> db;
  db.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    const unsigned char b =
        static_cast<unsigned char>((i * 13u + 7u) & 0xFFu);
    db.push_back(base64_encode(&b, 1));
  }

  // Deterministic spread of query indices across [0, N).
  std::vector<uint64_t> idxs;
  idxs.reserve(queries);
  std::mt19937_64 rng(/*seed=*/0xF0DBE);
  for (uint64_t q = 0; q < queries; ++q) {
    idxs.push_back(rng() % n);
  }

  const auto t_total_start = std::chrono::steady_clock::now();

  // ---- Setup: construct the Shard (Database + BaseParams). ----
  // This is the dominant one-time cost per DB; for FrodoPIR it
  // includes the Database::New decode pass + BaseParams::New, which
  // runs GenerateLweMatrixFromSeed once (m x dim) and computes RHS as
  // db * A (m * dim u32 wrapping multiplies per Database column).
  primihub::pir::frodo::Shard shard;
  std::string err;
  const auto t_setup_start = std::chrono::steady_clock::now();
  auto rc = primihub::pir::frodo::Shard::FromBase64Strings(
      db, lwe_dim, /*m=*/n, /*elem_size=*/8, plaintext_bits, &shard, &err);
  const double setup_ms = ms_since(t_setup_start);
  if (rc != primihub::retcode::SUCCESS) {
    if (csv) {
      std::printf("%llu,NA,NA,NA,setup_failed\n",
                  static_cast<unsigned long long>(n));
    } else {
      std::fprintf(stderr, "Shard::FromBase64Strings failed: %s\n",
                   err.c_str());
    }
    return 1;
  }
  const auto& bp = shard.GetBaseParams();
  const auto cp = primihub::pir::frodo::CommonParams::FromBaseParams(bp);

  // ---- Per-query loop: QueryParams::New + GenerateQuery + Respond
  // + ParseOutputAsBase64. ----
  double query_total_ms = 0.0;
  for (uint64_t q = 0; q < queries; ++q) {
    const auto t_q_start = std::chrono::steady_clock::now();
    primihub::pir::frodo::QueryParams qp;
    rc = primihub::pir::frodo::QueryParams::New(cp, bp, &qp, &err);
    if (rc != primihub::retcode::SUCCESS) {
      if (csv) {
        std::printf("%llu,%.3f,NA,NA,qp_new_failed\n",
                    static_cast<unsigned long long>(n), setup_ms);
      } else {
        std::fprintf(stderr, "QueryParams::New failed at q=%llu: %s\n",
                     static_cast<unsigned long long>(q), err.c_str());
      }
      return 1;
    }
    primihub::pir::frodo::Query query;
    rc = qp.GenerateQuery(idxs[q], &query, &err);
    if (rc != primihub::retcode::SUCCESS) {
      if (csv) {
        std::printf("%llu,%.3f,NA,NA,gen_query_failed\n",
                    static_cast<unsigned long long>(n), setup_ms);
      } else {
        std::fprintf(stderr, "GenerateQuery failed at q=%llu: %s\n",
                     static_cast<unsigned long long>(q), err.c_str());
      }
      return 1;
    }
    primihub::pir::frodo::Response resp;
    rc = shard.Respond(query, &resp, &err);
    if (rc != primihub::retcode::SUCCESS) {
      if (csv) {
        std::printf("%llu,%.3f,NA,NA,respond_failed\n",
                    static_cast<unsigned long long>(n), setup_ms);
      } else {
        std::fprintf(stderr, "Respond failed at q=%llu: %s\n",
                     static_cast<unsigned long long>(q), err.c_str());
      }
      return 1;
    }
    volatile auto sink = resp.ParseOutputAsBase64(qp);
    (void)sink;
    query_total_ms += ms_since(t_q_start);
  }

  const double per_query_ms =
      query_total_ms / static_cast<double>(queries);
  const double wall_ms = ms_since(t_total_start);

  if (csv) {
    // CSV schema: n,setup_ms,per_query_ms,wall_ms,status
    std::printf("%llu,%.3f,%.3f,%.3f,ok\n",
                static_cast<unsigned long long>(n), setup_ms,
                per_query_ms, wall_ms);
  } else {
    std::printf("FrodoPIR latency @ N=%llu, Q=%llu, dim=%zu, pt_bits=%zu:\n",
                static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(queries),
                lwe_dim, plaintext_bits);
    std::printf("  setup        ~ %.3f ms  (one-time per DB)\n", setup_ms);
    std::printf("  per_query    ~ %.3f ms  (avg over %llu queries)\n",
                per_query_ms,
                static_cast<unsigned long long>(queries));
    std::printf("  wall_total   ~ %.3f ms  (Setup + %llu queries)\n",
                wall_ms,
                static_cast<unsigned long long>(queries));
  }
  return 0;
}
