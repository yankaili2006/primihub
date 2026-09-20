/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * PIR operator benchmark: pirana vs frodo_pir vs id_pir on identical
 * logical DB workloads (single-process OnExecute timing). Not a
 * gtest — a standalone binary runnable on the bench host:
 *
 *   bazel build --config=linux_x86_64 \
 *       //src/primihub/kernel/pir/bench:pir_operator_bench
 *   bazel-bin/src/primihub/kernel/pir/bench/pir_operator_bench \
 *       [--n 4096] [--payload 32] [--iters 3]
 *
 * Each algorithm gets its native input encoding for the same logical
 * DB (deterministic bytes); we time full OnExecute (includes any
 * per-call setup — that asymmetry is inherent to the schemes and is
 * reported as-is).
 */
#include <glog/logging.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_pir.h"
#include "src/primihub/kernel/pir/operator/id_pir.h"
#include "src/primihub/kernel/pir/operator/pirana_pir/pirana_pir.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace {

using primihub::Role;
using primihub::retcode;
using primihub::pir::Options;
using primihub::pir::PirDataType;

Options MinimalOptions(const char* code) {
  Options o;
  o.code = code;
  o.role = Role::CLIENT;
  return o;
}

// Deterministic logical DB: payload i, byte b = (i*31 + b*7 + 5) & 0xFF.
std::vector<std::string> MakeDb(std::size_t n, std::size_t payload) {
  std::vector<std::string> db;
  db.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::string p(payload, '\0');
    for (std::size_t b = 0; b < payload; ++b) {
      p[b] = static_cast<char>((i * 31u + b * 7u + 5u) & 0xFFu);
    }
    db.push_back(p);
  }
  return db;
}

struct Timing {
  double total_ms = 0;
  bool ok = false;
};

Timing TimeOnExecute(const std::string& name, primihub::pir::BasePirOperator* op,
                     const PirDataType& input) {
  Timing t;
  PirDataType result;
  const auto start = std::chrono::steady_clock::now();
  const auto rc = op->OnExecute(input, &result);
  const auto end = std::chrono::steady_clock::now();
  if (rc != retcode::SUCCESS) {
    std::cout << name << ": OnExecute FAILED (skipped)\n";
    return t;
  }
  t.total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  t.ok = true;
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 3;  // ERROR only — keep stdout clean

  std::size_t n = 4096;
  std::size_t payload = 32;
  int iters = 3;
  std::string which = "all";
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::strtoul(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--payload") && i + 1 < argc) payload = std::strtoul(argv[++i], nullptr, 10);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--algo") && i + 1 < argc) which = argv[++i];
  }
  const bool run_pirana = which == "all" || which == "pirana";
  const bool run_frodo = which == "all" || which == "frodo";
  const bool run_id = which == "all" || which == "id";

  const std::string idx = std::to_string(n / 3);
  std::cout << "DB: " << n << " payloads x " << payload
            << " B, single query idx=" << idx << ", iters=" << iters << "\n\n";
  std::printf("%-12s %12s %15s %15s\n", "algorithm", "status",
              "setup+query ms", "best of iters");

  // ---- pirana (raw byte strings) ----
  if (run_pirana) {
    const auto db = MakeDb(n, payload);
    primihub::pir::PiranaPirOperator op(MinimalOptions("pirana_bench"));
    double best = 1e18;
    bool ok = false;
    for (int it = 0; it < iters; ++it) {
      PirDataType in;
      in["db_content"] = db;
      in["query_indices"] = {idx};
      auto t = TimeOnExecute("pirana", &op, in);
      if (t.ok) {
        best = std::min(best, t.total_ms);
        ok = true;
      }
    }
    std::printf("%-12s %12s %15s %15.1f\n", "pirana", ok ? "ok" : "FAIL", "-",
                ok ? best : -1.0);
  }

  // ---- frodo_pir (base64 uniform entries) ----
  if (run_frodo) {
    const auto db = MakeDb(n, payload);
    std::vector<std::string> b64;
    b64.reserve(db.size());
    for (const auto& p : db) {
      b64.push_back(p);  // raw bytes as std::string; Frodo takes base64 —
                         // use its operator contract: it base64-decodes, so
                         // pass base64 of the raw payload.
    }
    // FrodoPirOperator expects base64-encoded elements.
    for (auto& s : b64) {
      // encode in place via operator's expectation: reuse simple encoder
      static const char* kAlpha =
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
      std::string out;
      const auto* bytes = reinterpret_cast<const unsigned char*>(s.data());
      for (std::size_t i = 0; i < s.size(); i += 3) {
        unsigned v = bytes[i] << 16;
        if (i + 1 < s.size()) v |= bytes[i + 1] << 8;
        if (i + 2 < s.size()) v |= bytes[i + 2];
        out += kAlpha[(v >> 18) & 63];
        out += kAlpha[(v >> 12) & 63];
        out += (i + 1 < s.size()) ? kAlpha[(v >> 6) & 63] : '=';
        out += (i + 2 < s.size()) ? kAlpha[v & 63] : '=';
      }
      s = out;
    }
    primihub::pir::FrodoPirOperator op(MinimalOptions("frodo_bench"));
    double best = 1e18;
    bool ok = false;
    for (int it = 0; it < iters; ++it) {
      PirDataType in;
      in["db_content"] = b64;
      in["query_indices"] = {idx};
      auto t = TimeOnExecute("frodo_pir", &op, in);
      if (t.ok) {
        best = std::min(best, t.total_ms);
        ok = true;
      }
    }
    std::printf("%-12s %12s %15s %15.1f\n", "frodo_pir", ok ? "ok" : "FAIL",
                "-", ok ? best : -1.0);
  }

  // ---- id_pir (plaintext passthrough baseline) ----
  if (run_id) {
    const auto db = MakeDb(n, payload);
    primihub::pir::IdPirOperator op(MinimalOptions("id_pir_bench"));
    double best = 1e18;
    bool ok = false;
    for (int it = 0; it < iters; ++it) {
      PirDataType in;
      // IdPirOperator contract: map key -> list of "val" strings; it
      // echoes matching keys (plaintext lookup baseline).
      in["query"] = {idx};
      in["db_content"] = db;
      auto t = TimeOnExecute("id_pir", &op, in);
      if (t.ok) {
        best = std::min(best, t.total_ms);
        ok = true;
      }
    }
    std::printf("%-12s %12s %15s %15.1f\n", "id_pir", ok ? "ok" : "FAIL", "-",
                ok ? best : -1.0);
  }

  google::ShutdownGoogleLogging();
  return 0;
}
