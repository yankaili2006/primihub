/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * simple_pir_latency_bench — drives SimplePirOperator at caller-supplied
 * scale and prints per-stage wall-clock latency to stdout in CSV form.
 *
 * Usage:
 *   simple_pir_latency_bench --n 1024 --queries 16 [--csv]
 *
 *   --n N         number of byte entries in the synthetic DB (must yield
 *                 a sqrt that's a multiple of 8 — kRowAlignment guard in
 *                 SimplePirOperator). Recommended: perfect squares with
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
 * The bench wraps SimplePirOperator as-is: it does NOT bypass any of the
 * activation / row-alignment / Database::SetupShape guards, so failures
 * here mirror what production callers would see. Latency includes the
 * Init+Setup+SquishDB+per-query loop end-to-end.
 */
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir.h"

namespace {

using primihub::pir::SimplePirOperator;
using primihub::pir::Options;
using primihub::pir::PirDataType;
using primihub::Role;
using primihub::retcode;

double ms_since(const std::chrono::steady_clock::time_point& start) {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now() - start).count();
}

}  // namespace

struct OpTiming {
  double init_ms = 0.0;
  double setup_ms = 0.0;
  uint64_t queries = 0;
  double query_total_ms = 0.0;
  bool parsed = false;
  // -1 = unparsed / v1 operator without hint_hit; 0 = cache miss;
  // 1 = cache hit. Populated by the v2 regex.
  int hint_hit = -1;
};

// Drive op.OnExecute() with stderr captured to a tmpfile, then parse
// the structured "SimplePirOperator: timing init_ms=… setup_ms=…
// queries=… query_total_ms=…" LOG line. Falls back to total wall-time
// only if the line is absent (operator running without timing patch).
primihub::retcode ExecuteAndTime(SimplePirOperator* op,
                                  const PirDataType& input,
                                  PirDataType* result,
                                  OpTiming* timing,
                                  double* wall_ms) {
  // Redirect stderr (where glog writes) to a tmpfile for the duration
  // of OnExecute.
  std::fflush(stderr);
  char tmpl[] = "/tmp/simple_pir_bench_stderr_XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) {
    return op->OnExecute(input, result);
  }
  const int saved_stderr = dup(fileno(stderr));
  dup2(fd, fileno(stderr));

  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  auto rc = op->OnExecute(input, result);
  const auto t1 = clock::now();

  std::fflush(stderr);
  dup2(saved_stderr, fileno(stderr));
  close(saved_stderr);
  close(fd);

  *wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::ifstream f(tmpl);
  std::stringstream ss;
  ss << f.rdbuf();
  std::remove(tmpl);
  const std::string captured = ss.str();

  // Match either the v1 timing line (no hint_hit field) or the v2
  // line that includes hint_hit. The v2 form is emitted after the
  // HintCache landing — we tolerate both to keep this binary
  // compatible with older operator builds.
  // SimplePIR's structured timing line includes squish_ms (SimplePIR
  // exposes it separately, unlike DoublePIR which folds it into setup).
  // The bench parses it but doesn't currently distinguish it in output —
  // setup_ms here means "Init+Setup", as in the upstream paper.
  std::regex re(R"(SimplePirOperator: timing init_ms=([\d.eE+-]+) setup_ms=([\d.eE+-]+) squish_ms=([\d.eE+-]+) hint_hit=(\d+) queries=(\d+) query_total_ms=([\d.eE+-]+))");
  std::smatch m;
  if (std::regex_search(captured, m, re) && m.size() == 7) {
    timing->init_ms = std::stod(m[1]);
    timing->setup_ms = std::stod(m[2]) + std::stod(m[3]);
    timing->hint_hit = std::stoi(m[4]);
    timing->queries = std::stoull(m[5]);
    timing->query_total_ms = std::stod(m[6]);
    timing->parsed = true;
  }
  return rc;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  uint64_t n = 1024;
  uint64_t queries = 16;
  bool csv = false;
  std::string hint_path;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--n" && i + 1 < argc) {
      n = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--queries" && i + 1 < argc) {
      queries = std::strtoull(argv[++i], nullptr, 10);
    } else if (a == "--hint-path" && i + 1 < argc) {
      hint_path = argv[++i];
    } else if (a == "--csv") {
      csv = true;
    } else if (a == "-h" || a == "--help") {
      std::fprintf(stderr,
                   "Usage: %s --n N --queries Q [--csv] "
                   "[--hint-path /path/to/hint_cache.bin]\n",
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
  opt.hint_path = hint_path;  // empty string = persistence disabled
  SimplePirOperator op(opt);

  // Single OnExecute with all queries. Capture per-stage timing from
  // the operator's structured LOG line (chrono-instrumented in
  // double_pir.cc as of commit f08a45*). Falls back to wall-time-only
  // if the LOG line is absent.
  PirDataType in;
  in["db_content"] = db;
  in["query_indices"] = idxs;
  PirDataType out;
  OpTiming t;
  double wall_ms = 0.0;
  auto rc = ExecuteAndTime(&op, in, &out, &t, &wall_ms);
  if (rc != retcode::SUCCESS) {
    if (csv) std::printf("%llu,NA,NA,NA,queries_failed\n",
                         static_cast<unsigned long long>(n));
    else std::fprintf(stderr, "FAILED execution: see logs above\n");
    return 1;
  }

  double init_ms;
  double setup_ms;
  double per_query_ms;
  if (t.parsed && t.queries > 0) {
    init_ms = t.init_ms;
    setup_ms = t.setup_ms;
    per_query_ms = t.query_total_ms / static_cast<double>(t.queries);
  } else {
    // Fallback path — no per-stage detail available.
    init_ms = 0.0;
    setup_ms = wall_ms;  // worst case attribute everything to setup
    per_query_ms = 0.0;
  }
  const double p50 = per_query_ms;
  const double pmax = per_query_ms;

  if (csv) {
    // CSV schema (v3): n,init_ms,setup_ms,per_query_ms,wall_ms,hint_hit,status
    // hint_hit = -1 when timing line wasn't parsed (v1 operator) or
    // wasn't emitted (no persistence integration).
    std::printf("%llu,%.3f,%.3f,%.3f,%.3f,%d,ok\n",
                static_cast<unsigned long long>(n), init_ms, setup_ms,
                per_query_ms, wall_ms, t.hint_hit);
  } else {
    std::printf("SimplePIR latency @ N=%llu, Q=%llu:\n",
                static_cast<unsigned long long>(n),
                static_cast<unsigned long long>(queries));
    std::printf("  init         ~ %.3f ms  (matrix sampling)\n",  init_ms);
    std::printf("  setup        ~ %.3f ms  (one-time per DB)\n",  setup_ms);
    std::printf("  per_query    ~ %.3f ms  (avg over %llu queries)\n",
                per_query_ms,
                static_cast<unsigned long long>(queries));
    std::printf("  wall_total   ~ %.3f ms  (Init + Setup + %llu queries)\n",
                wall_ms,
                static_cast<unsigned long long>(queries));
    if (t.hint_hit >= 0) {
      std::printf("  hint_hit     = %d  (%s)\n", t.hint_hit,
                  t.hint_hit ? "warm-start, Setup skipped"
                             : "cold-start, Setup ran");
    }
  }
  // p50 / pmax suppression-warning silencer.
  (void)p50;
  (void)pmax;
  return 0;
}
