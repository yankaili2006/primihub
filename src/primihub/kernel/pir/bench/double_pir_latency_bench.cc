/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * double_pir_latency_bench — drives DoublePirOperator at caller-supplied
 * scale and prints per-stage wall-clock latency to stdout in CSV form.
 *
 * Usage:
 *   double_pir_latency_bench --n 1024 --queries 16 [--csv]
 *
 *   --n N         number of byte entries in the synthetic DB (must yield
 *                 a sqrt that's a multiple of 8 — kRowAlignment guard in
 *                 DoublePirOperator). Recommended: perfect squares with
 *                 sqrt % 8 == 0, e.g. 64, 256, 1024, 4096, 16384, ...
 *   --queries Q   number of queries to issue (used for the per-query
 *                 latency p50/max). Q indices are picked deterministically
 *                 across [0, N).
 *   --csv         emit a single CSV line "n,setup_ms,query_p50_ms,query_max_ms,ok"
 *                 (default: a human-readable block).
 *
 * The bench drives the operator once per N: a single Setup call is
 * amortized across all queries. Setup + per-query times are measured
 * separately via two OnExecute invocations (one Setup-only flavor would
 * need an internal hook we don\'t add for the bench).
 *
 * The bench wraps DoublePirOperator as-is: it does NOT bypass any of the
 * activation / row-alignment / Database::SetupShape guards, so failures
 * here mirror what production callers would see. Latency includes the
 * Init+Setup+SquishDB+per-query loop end-to-end.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"

namespace {

using primihub::pir::DoublePirOperator;
using primihub::pir::Options;
using primihub::pir::PirDataType;
using primihub::Role;
using primihub::retcode;

double ms_since(const std::chrono::steady_clock::time_point& start) {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  uint64_t n = 1024;
  uint64_t queries = 16;
  bool csv = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n" && i + 1 < argc) {
      n = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--queries" && i + 1 < argc) {
      queries = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--csv") {
      csv = true;
    } else if (a == "-h" || a == "--help") {
      std::fprintf(stderr,
                   "Usage: %s --n N --queries Q [--csv]\n",
                   argv[0]);
      return 0;
    }
  }
  if (queries == 0 || n == 0) {
    std::fprintf(stderr, "n and queries must be > 0\n");
    return 2;
  }

  // Build a deterministic byte DB.
  std::vector<std::string> db;
  db.reserve(n);
  for (uint64_t i = 0; i < n; ++i) {
    db.push_back(std::to_string((i * 13 + 7) & 0xFF));
  }
  // Deterministic spread of query indices across [0, N).
  std::vector<std::string> idxs;
  idxs.reserve(queries);
  std::mt19937_64 rng(/*seed=*/0xD0DBE);
  for (uint64_t q = 0; q < queries; ++q) {
    const uint64_t pick = rng() % n;
    idxs.push_back(std::to_string(pick));
  }

  Options opt;
  opt.self_party = "bench";
  opt.role = Role::CLIENT;
  DoublePirOperator op(opt);

  // First call with a 1-query input to measure Setup amortization.
  PirDataType setup_in;
  setup_in["db_content"] = db;
  setup_in["query_indices"] = {idxs[0]};
  PirDataType setup_out;
  auto t0 = std::chrono::steady_clock::now();
  auto rc1 = op.OnExecute(setup_in, &setup_out);
  const double setup_plus_one_ms = ms_since(t0);
  if (rc1 != retcode::SUCCESS) {
    if (csv) std::printf("%llu,NA,NA,NA,setup_failed\n",
                         static_cast<unsigned long long>(n));
    else
      std::fprintf(stderr, "FAILED setup: see logs above\n");
    return 1;
  }

  // Second call with all queries — Setup repeats, but we subtract the
  // single-query timing from the multi-query timing to approximate
  // per-query latency. (DoublePirOperator does not currently expose a
  // hint-cache; every OnExecute reruns Init+Setup. The math works
  // because per-Setup costs are stable across calls with the same DB.)
  PirDataType all_in;
  all_in["db_content"] = db;
  all_in["query_indices"] = idxs;
  PirDataType all_out;
  t0 = std::chrono::steady_clock::now();
  auto rc2 = op.OnExecute(all_in, &all_out);
  const double setup_plus_n_ms = ms_since(t0);
  if (rc2 != retcode::SUCCESS) {
    if (csv) std::printf("%llu,NA,NA,NA,queries_failed\n",
                         static_cast<unsigned long long>(n));
    else
      std::fprintf(stderr, "FAILED multi-query: see logs above\n");
    return 1;
  }

  // Approximations:
  //   setup_ms  ~ setup_plus_one_ms - per_query_ms
  //   per_query ~ (setup_plus_n_ms - setup_plus_one_ms) / (queries - 1)
  double per_query_ms = 0.0;
  if (queries > 1) {
    per_query_ms = (setup_plus_n_ms - setup_plus_one_ms) /
                   static_cast<double>(queries - 1);
  } else {
    per_query_ms = setup_plus_one_ms;
  }
  const double setup_ms = setup_plus_one_ms - per_query_ms;
  // We only have two data points so p50 and max collapse to per_query_ms.
  // Future work: log per-query times inside OnExecute for true p50/max.
  const double p50 = per_query_ms;
  const double pmax = per_query_ms;

  if (csv) {
    std::printf("%llu,%.3f,%.3f,%.3f,ok\n",
                static_cast<unsigned long long>(n), setup_ms, p50, pmax);
  } else {
    std::printf("DoublePIR latency @ N=%llu, Q=%llu:\n",
                static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(queries));
    std::printf("  setup        ~ %.3f ms  (one-time per DB)\n", setup_ms);
    std::printf("  per_query    ~ %.3f ms  (avg over %llu queries)\n",
                per_query_ms,
                static_cast<unsigned long long>(queries - 1));
    std::printf("  total        ~ %.3f ms  (1 setup + %llu queries)\n",
                setup_plus_n_ms,
                static_cast<unsigned long long>(queries));
  }
  return 0;
}
