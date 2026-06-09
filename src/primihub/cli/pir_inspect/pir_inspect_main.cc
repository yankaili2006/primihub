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
#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"
#include "src/primihub/kernel/pir/operator/simple_pir/hint_serialize.h"

#include <cerrno>
#include <fstream>
#include <sstream>

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
    "                   assume-non-colluding=true latency-budget=ms\n"
    "\n"
    "  pir_inspect cache <path> [--emit-csv]\n"
    "      Dump on-disk HintCache summary or CSV. Auto-detects PHHC\n"
    "      (DoublePIR) vs PSHC (SimplePIR) via the file's magic.\n";
}

}  // namespace

// --- pir_inspect cache <path> -------------------------------------------
//
// Reads a PHHC HintCache file (written by HintCache::SaveToFile, task 5.6
// chunks 4+5) and prints a human-readable summary. Useful for verifying
// what's been persisted before restarting a production node, and for
// debugging "why didn't my hint hit the cache" investigations.
//
// Output:
//   - default: a multi-line per-entry summary (fingerprint hex, blob
//     bytes, matrix shapes, total cell count, DBinfo highlights)
//   - --emit-csv: one CSV row per entry — for shell pipelines / monitoring

namespace {

// Two cache file magics — distinct so SimplePIR and DoublePIR caches
// can co-exist in the same directory and pir_inspect dispatches at
// parse time. CLI keeps its own copy of the constants so it doesn't
// need to drag the storage-layer kCacheMagic out of an anonymous
// namespace.
constexpr char kDoublePirCacheMagic[4] = {'P', 'H', 'H', 'C'};
constexpr char kSimplePirCacheMagic[4] = {'P', 'S', 'H', 'C'};
constexpr uint16_t kCacheVersionCli = 1;

inline bool ReadU16LE(const std::string& s, size_t* off, uint16_t* out) {
  if (*off + 2 > s.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data() + *off);
  *out = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  *off += 2;
  return true;
}

inline bool ReadU64LE(const std::string& s, size_t* off, uint64_t* out) {
  if (*off + 8 > s.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data() + *off);
  *out = 0;
  for (int i = 0; i < 8; ++i) {
    *out |= static_cast<uint64_t>(p[i]) << (i * 8);
  }
  *off += 8;
  return true;
}

// Reads the file, validates magic + version + count, and returns the
// remainder of the buffer in *out_payload starting just after the
// count u64. Algorithm-specific helpers then walk that payload.
//
// Returns 0 on success, sets *algo to either "double_pir" or
// "simple_pir". Returns 1 on parse failure with stderr populated.
int ReadCacheHeader(const std::string& path, std::string* data_out,
                    std::string* algo, uint16_t* version_out,
                    uint64_t* count_out, size_t* payload_off_out) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "pir_inspect cache: cannot open " << path << ": "
              << std::strerror(errno) << "\n";
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *data_out = ss.str();
  const std::string& data = *data_out;

  if (data.size() < 4) {
    std::cerr << "pir_inspect cache: " << path << " shorter than magic\n";
    return 1;
  }
  if (std::memcmp(data.data(), kDoublePirCacheMagic, 4) == 0) {
    *algo = "double_pir";
  } else if (std::memcmp(data.data(), kSimplePirCacheMagic, 4) == 0) {
    *algo = "simple_pir";
  } else {
    std::cerr << "pir_inspect cache: " << path
              << " is not a HintCache file (magic is not PHHC or PSHC)\n";
    return 1;
  }
  size_t off = 4;
  uint16_t version = 0, reserved = 0;
  if (!ReadU16LE(data, &off, &version) ||
      !ReadU16LE(data, &off, &reserved)) {
    std::cerr << "pir_inspect cache: " << path << " truncated header\n";
    return 1;
  }
  if (version != kCacheVersionCli) {
    std::cerr << "pir_inspect cache: " << path
              << " has unsupported version " << version
              << " (this build understands " << kCacheVersionCli << ")\n";
    return 1;
  }
  uint64_t count = 0;
  if (!ReadU64LE(data, &off, &count)) {
    std::cerr << "pir_inspect cache: " << path
              << " truncated entry count\n";
    return 1;
  }
  *version_out = version;
  *count_out = count;
  *payload_off_out = off;
  return 0;
}

int CmdCacheDouble(const std::string& path, const std::string& data,
                   uint16_t version, uint64_t count, size_t off,
                   bool emit_csv) {
  if (emit_csv) {
    std::cout
        << "algorithm,entry_idx,fp_hex,blob_bytes,"
        << "A1_rows,A1_cols,A2_rows,A2_cols,"
        << "H1sq_rows,H1sq_cols,A2copyT_rows,A2copyT_cols,"
        << "H2msg_rows,H2msg_cols,total_cells,"
        << "info_num,info_p,info_logq\n";
  } else {
    std::cout << "PHHC HintCache (DoublePIR) @ " << path << "\n"
              << "  version : " << version << "\n"
              << "  entries : " << count << "\n"
              << "  bytes   : " << data.size() << "\n\n";
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t fp = 0, blob_len = 0;
    if (!ReadU64LE(data, &off, &fp) || !ReadU64LE(data, &off, &blob_len)) {
      std::cerr << "pir_inspect cache: truncated entry header at index "
                << i << "\n";
      return 1;
    }
    if (off + blob_len > data.size()) {
      std::cerr << "pir_inspect cache: truncated entry body at index "
                << i << "\n";
      return 1;
    }
    const std::string blob = data.substr(off, blob_len);
    off += blob_len;
    primihub::pir::double_pir::DoublePirHint hint;
    std::string blob_err;
    if (primihub::pir::double_pir::DeserializeHint(blob, &hint, &blob_err) !=
        primihub::retcode::SUCCESS) {
      std::cerr << "pir_inspect cache: entry " << i
                << " DeserializeHint failed: " << blob_err << "\n";
      return 1;
    }
    const uint64_t total_cells =
        hint.A1.size() + hint.A2.size() + hint.H1_squished.size() +
        hint.A2_copy_transposed.size() + hint.H2_msg.size();
    if (emit_csv) {
      std::cout << "double_pir," << i << ",0x" << std::hex << fp
                << std::dec << "," << blob_len << ","
                << hint.A1.rows() << "," << hint.A1.cols() << ","
                << hint.A2.rows() << "," << hint.A2.cols() << ","
                << hint.H1_squished.rows() << "," << hint.H1_squished.cols()
                << ","
                << hint.A2_copy_transposed.rows() << ","
                << hint.A2_copy_transposed.cols() << ","
                << hint.H2_msg.rows() << "," << hint.H2_msg.cols() << ","
                << total_cells << ","
                << hint.info_after_setup.num << ","
                << hint.info_after_setup.p << ","
                << hint.info_after_setup.logq << "\n";
    } else {
      std::cout << "entry[" << i << "] fp=0x" << std::hex << fp << std::dec
                << " blob=" << blob_len << " B"
                << " | A1=" << hint.A1.rows() << "x" << hint.A1.cols()
                << " A2=" << hint.A2.rows() << "x" << hint.A2.cols()
                << " H1sq=" << hint.H1_squished.rows() << "x"
                << hint.H1_squished.cols()
                << " A2copyT=" << hint.A2_copy_transposed.rows() << "x"
                << hint.A2_copy_transposed.cols()
                << " H2msg=" << hint.H2_msg.rows() << "x"
                << hint.H2_msg.cols()
                << " | cells=" << total_cells
                << " info{num=" << hint.info_after_setup.num
                << " p=" << hint.info_after_setup.p
                << " logq=" << hint.info_after_setup.logq << "}\n";
    }
  }
  if (off != data.size()) {
    std::cerr << "pir_inspect cache: " << (data.size() - off)
              << " trailing bytes after last entry\n";
    return 1;
  }
  return 0;
}

int CmdCacheSimple(const std::string& path, const std::string& data,
                   uint16_t version, uint64_t count, size_t off,
                   bool emit_csv) {
  if (emit_csv) {
    std::cout
        << "algorithm,entry_idx,fp_hex,blob_bytes,"
        << "A_rows,A_cols,H_rows,H_cols,total_cells,"
        << "info_num,info_p,info_logq\n";
  } else {
    std::cout << "PSHC HintCache (SimplePIR) @ " << path << "\n"
              << "  version : " << version << "\n"
              << "  entries : " << count << "\n"
              << "  bytes   : " << data.size() << "\n\n";
  }
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t fp = 0, blob_len = 0;
    if (!ReadU64LE(data, &off, &fp) || !ReadU64LE(data, &off, &blob_len)) {
      std::cerr << "pir_inspect cache: truncated entry header at index "
                << i << "\n";
      return 1;
    }
    if (off + blob_len > data.size()) {
      std::cerr << "pir_inspect cache: truncated entry body at index "
                << i << "\n";
      return 1;
    }
    const std::string blob = data.substr(off, blob_len);
    off += blob_len;
    primihub::pir::simple_pir::SimplePirHint hint;
    std::string blob_err;
    if (primihub::pir::simple_pir::DeserializeHint(blob, &hint, &blob_err) !=
        primihub::retcode::SUCCESS) {
      std::cerr << "pir_inspect cache: entry " << i
                << " DeserializeHint failed: " << blob_err << "\n";
      return 1;
    }
    const uint64_t total_cells = hint.A.size() + hint.H.size();
    if (emit_csv) {
      std::cout << "simple_pir," << i << ",0x" << std::hex << fp
                << std::dec << "," << blob_len << ","
                << hint.A.rows() << "," << hint.A.cols() << ","
                << hint.H.rows() << "," << hint.H.cols() << ","
                << total_cells << ","
                << hint.info_after_squish.num << ","
                << hint.info_after_squish.p << ","
                << hint.info_after_squish.logq << "\n";
    } else {
      std::cout << "entry[" << i << "] fp=0x" << std::hex << fp << std::dec
                << " blob=" << blob_len << " B"
                << " | A=" << hint.A.rows() << "x" << hint.A.cols()
                << " H=" << hint.H.rows() << "x" << hint.H.cols()
                << " | cells=" << total_cells
                << " info{num=" << hint.info_after_squish.num
                << " p=" << hint.info_after_squish.p
                << " logq=" << hint.info_after_squish.logq << "}\n";
    }
  }
  if (off != data.size()) {
    std::cerr << "pir_inspect cache: " << (data.size() - off)
              << " trailing bytes after last entry\n";
    return 1;
  }
  return 0;
}

// Dispatcher — sniffs the file's 4-byte magic and routes to the
// algorithm-specific cache reader. Returns 0 on success / 1 on any
// parse failure (stderr already populated by the helpers).
int CmdCache(const std::string& path, bool emit_csv) {
  std::string data;
  std::string algo;
  uint16_t version = 0;
  uint64_t count = 0;
  size_t payload_off = 0;
  int rc = ReadCacheHeader(path, &data, &algo, &version, &count,
                            &payload_off);
  if (rc != 0) return rc;
  if (algo == "double_pir") {
    return CmdCacheDouble(path, data, version, count, payload_off, emit_csv);
  }
  if (algo == "simple_pir") {
    return CmdCacheSimple(path, data, version, count, payload_off, emit_csv);
  }
  // ReadCacheHeader guards this — defensive fallthrough.
  std::cerr << "pir_inspect cache: unknown algorithm '" << algo << "'\n";
  return 1;
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
  if (sub == "cache") {
    if (argc < 3) {
      std::cerr << "usage: pir_inspect cache <path> [--emit-csv]\n";
      return 2;
    }
    bool emit_csv = false;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--emit-csv") emit_csv = true;
    }
    return CmdCache(argv[2], emit_csv);
  }
  if (sub == "--help" || sub == "-h" || sub == "help") {
    Usage();
    return 0;
  }
  std::cerr << "unknown subcommand: " << sub << "\n";
  Usage();
  return 2;
}
