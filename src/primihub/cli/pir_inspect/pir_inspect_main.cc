/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Standalone CLI for inspecting the multi-algorithm PIR framework.
 * Does NOT execute PIR queries — those go through primihub-node + primihub-cli.
 * Use cases:
 *   pir_inspect list                            — list registered algorithms
 *   pir_inspect caps <algorithm>                — dump capabilities JSON
 *   pir_inspect auto --db-size N --query-type T — selector dry-run, ranked
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/selector.h"

using primihub::pir::AlgoMatch;
using primihub::pir::Backend;
using primihub::pir::Constraints;
using primihub::pir::LatencyBudget;
using primihub::pir::PirRegistry;
using primihub::pir::PirSelector;
using primihub::pir::QueryType;
using primihub::pir::ThreatModel;
using primihub::pir::ToString;

namespace {

QueryType ParseQueryType(const std::string& s) {
  if (s == "index") return QueryType::Index;
  if (s == "keyword") return QueryType::Keyword;
  if (s == "semantic") return QueryType::Semantic;
  std::cerr << "unknown query type: " << s
            << " (expected index|keyword|semantic)\n";
  std::exit(2);
}

LatencyBudget ParseBudget(const std::string& s) {
  if (s == "any") return LatencyBudget::Any;
  if (s == "seconds") return LatencyBudget::Seconds;
  if (s == "sub-second" || s == "subsecond") return LatencyBudget::SubSecond;
  if (s == "ms") return LatencyBudget::Ms;
  std::cerr << "unknown latency budget: " << s
            << " (expected any|seconds|sub-second|ms)\n";
  std::exit(2);
}

ThreatModel ParseThreatModel(const std::string& s) {
  if (s == "semi-honest") return ThreatModel::SemiHonest;
  if (s == "semi-honest-non-colluding") return ThreatModel::SemiHonestNonColluding;
  if (s == "malicious") return ThreatModel::Malicious;
  std::cerr << "unknown threat model: " << s << "\n";
  std::exit(2);
}

uint64_t ParseSize(const std::string& s) {
  if (s.empty()) return 0;
  size_t end = 0;
  // Accept scientific notation like 1e8
  double d = std::stod(s, &end);
  if (end == 0) return 0;
  return static_cast<uint64_t>(d);
}

void PrintList() {
  auto algos = PirRegistry::Instance().ListAlgorithms();
  std::printf("%-20s | %-15s | %-7s | %-5s | %-12s | %s\n",
              "algorithm", "query_types", "servers", "hint",
              "perf_class", "rec_max_db");
  std::printf("%s\n", std::string(85, '-').c_str());
  for (const auto& algo : algos) {
    const auto* c = PirRegistry::Instance().GetCapabilities(algo);
    if (!c) continue;
    std::string types;
    for (auto t : c->query_types) {
      if (!types.empty()) types.append(",");
      types.append(ToString(t));
    }
    std::string servers = std::to_string(c->min_servers);
    if (c->min_servers != c->max_servers) {
      servers.append("-").append(std::to_string(c->max_servers));
    }
    std::printf("%-20s | %-15s | %-7s | %-5s | %-12s | %lu\n",
                algo.c_str(), types.c_str(), servers.c_str(),
                c->needs_preprocess ? "yes" : "no",
                ToString(c->perf_class),
                static_cast<unsigned long>(c->recommended_max_db_size));
  }
  std::printf("\nTotal: %zu algorithm(s) registered\n", algos.size());
}

int CmdCapabilities(const std::string& algo) {
  const auto* c = PirRegistry::Instance().GetCapabilities(algo);
  if (!c) {
    std::cerr << "algorithm not registered: " << algo << "\n";
    return 1;
  }
  std::printf("%s\n", c->ToJson().c_str());
  return 0;
}

int CmdAuto(const std::map<std::string, std::string>& flags) {
  Constraints c;
  auto get = [&](const std::string& k, const std::string& def) {
    auto it = flags.find(k);
    return it != flags.end() ? it->second : def;
  };
  c.db_size = ParseSize(get("db-size", "0"));
  c.element_size = ParseSize(get("element-size", "0"));
  c.query_type = ParseQueryType(get("query-type", "index"));
  c.allow_two_server = get("allow-two-server", "0") == "1" ||
                       get("allow-two-server", "false") == "true";
  c.client_can_cache_hint =
      get("client-can-cache-hint", "0") == "1" ||
      get("client-can-cache-hint", "false") == "true";
  c.assume_non_colluding = get("assume-non-colluding", "0") == "1" ||
                           get("assume-non-colluding", "false") == "true";
  c.latency_budget = ParseBudget(get("latency-budget", "any"));
  c.min_threat_model = ParseThreatModel(get("min-threat-model", "semi-honest"));
  c.bandwidth_priority = get("bandwidth-priority", "0") == "1" ||
                         get("bandwidth-priority", "false") == "true";
  c.include_skeletons = get("include-skeletons", "0") == "1" ||
                        get("include-skeletons", "false") == "true";

  bool dry_run = flags.count("dry-run") &&
                 (flags.at("dry-run") == "1" || flags.at("dry-run") == "true" ||
                  flags.at("dry-run").empty());

  auto matches = PirSelector{}.RecommendWithRationale(c);
  if (dry_run) {
    std::printf("%-20s | %-6s | %-9s | %-7s | %s\n",
                "algorithm", "passes", "score", "comm_KB", "fail_reasons");
    std::printf("%s\n", std::string(90, '-').c_str());
    for (const auto& m : matches) {
      std::string fails;
      for (const auto& f : m.failed_checks) {
        if (!fails.empty()) fails.append("; ");
        fails.append(f);
      }
      std::printf("%-20s | %-6s | %-9lu | %-7lu | %s\n",
                  m.algorithm.c_str(),
                  m.passes ? "yes" : "no",
                  static_cast<unsigned long>(m.score),
                  static_cast<unsigned long>(m.typical_query_comm_bytes / 1024),
                  fails.c_str());
    }
    return 0;
  }
  // Non-dry: just print the winning algorithm name (machine-readable)
  for (const auto& m : matches) {
    if (m.passes) {
      std::printf("%s\n", m.algorithm.c_str());
      return 0;
    }
  }
  std::cerr << "no algorithm satisfies constraints\n";
  return 1;
}

void Usage() {
  std::cerr <<
    "Usage:\n"
    "  pir_inspect list\n"
    "      List all registered PIR algorithms with their capability summary.\n"
    "\n"
    "  pir_inspect caps <algorithm>\n"
    "      Dump full capabilities JSON for one algorithm.\n"
    "\n"
    "  pir_inspect auto [flags]\n"
    "      Run the selector. Flags (key=value):\n"
    "        db-size=<N>                  e.g. 1e8\n"
    "        element-size=<bytes>\n"
    "        query-type=<index|keyword|semantic>\n"
    "        allow-two-server=<true|false>\n"
    "        client-can-cache-hint=<true|false>\n"
    "        assume-non-colluding=<true|false>\n"
    "        latency-budget=<any|seconds|sub-second|ms>\n"
    "        min-threat-model=<semi-honest|semi-honest-non-colluding|malicious>\n"
    "        bandwidth-priority=<true|false>\n"
    "        include-skeletons=<true|false>  include kIsSkeleton algos in recs\n"
    "        dry-run=<true|false>          show ranked table with rationale\n"
    "\n"
    "Examples:\n"
    "  pir_inspect list\n"
    "  pir_inspect caps id_pir\n"
    "  pir_inspect auto db-size=1e8 query-type=index allow-two-server=true \\\n"
    "                   assume-non-colluding=true latency-budget=ms\n";
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  PirRegistry::EnsureRegistered();

  if (argc < 2) {
    Usage();
    return 2;
  }
  std::string sub = argv[1];
  if (sub == "list") {
    PrintList();
    return 0;
  }
  if (sub == "caps") {
    if (argc < 3) {
      std::cerr << "usage: pir_inspect caps <algorithm>\n";
      return 2;
    }
    return CmdCapabilities(argv[2]);
  }
  if (sub == "auto") {
    std::map<std::string, std::string> flags;
    for (int i = 2; i < argc; ++i) {
      std::string a = argv[i];
      auto eq = a.find('=');
      if (eq != std::string::npos) {
        flags[a.substr(0, eq)] = a.substr(eq + 1);
      } else {
        flags[a] = "";  // bare flag, treated as boolean true
      }
    }
    return CmdAuto(flags);
  }
  if (sub == "--help" || sub == "-h" || sub == "help") {
    Usage();
    return 0;
  }
  std::cerr << "unknown subcommand: " << sub << "\n";
  Usage();
  return 2;
}
