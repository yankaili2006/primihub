/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_CAPABILITIES_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_CAPABILITIES_H_

#include <cstdint>
#include <set>
#include <string>
#include <sstream>
#include "src/primihub/kernel/pir/common.h"

namespace primihub::pir {

// Machine-readable capability declaration for a PIR algorithm. Each algorithm
// registers an instance at static initialization; the Selector uses these to
// rank candidates and the CLI prints them via `pir capabilities <algo>`.
struct PirCapabilities {
  std::set<QueryType> query_types;
  uint32_t min_servers = 1;
  uint32_t max_servers = 1;
  bool needs_preprocess = false;
  bool hint_per_database = false;  // false = hint is per-client / no hint
  ThreatModel threat_model = ThreatModel::SemiHonest;
  PerfClass perf_class = PerfClass::Seconds;
  uint64_t recommended_max_db_size = 0;
  std::set<Backend> backends = {Backend::CPU};
  uint64_t typical_query_comm_bytes = 0;
  uint64_t typical_hint_size_bytes = 0;  // 0 = no hint

  // Self-consistency check used by tests and by Registry on register.
  // Returns empty string when consistent, otherwise a human-readable error.
  std::string Check() const {
    std::ostringstream os;
    if (min_servers == 0) {
      os << "min_servers must be >= 1";
      return os.str();
    }
    if (min_servers > max_servers) {
      os << "min_servers (" << min_servers
         << ") > max_servers (" << max_servers << ")";
      return os.str();
    }
    if (query_types.empty()) {
      os << "query_types must not be empty";
      return os.str();
    }
    if (!needs_preprocess && typical_hint_size_bytes != 0) {
      os << "typical_hint_size_bytes must be 0 when needs_preprocess=false";
      return os.str();
    }
    if (backends.empty()) {
      os << "backends must not be empty";
      return os.str();
    }
    return "";
  }

  std::string ToJson() const {
    std::ostringstream os;
    os << "{";
    os << "\"query_types\":[";
    bool first = true;
    for (auto q : query_types) {
      if (!first) os << ",";
      os << "\"" << ToString(q) << "\"";
      first = false;
    }
    os << "],";
    os << "\"min_servers\":" << min_servers << ",";
    os << "\"max_servers\":" << max_servers << ",";
    os << "\"needs_preprocess\":" << (needs_preprocess ? "true" : "false") << ",";
    os << "\"hint_per_database\":" << (hint_per_database ? "true" : "false") << ",";
    os << "\"threat_model\":\"" << ToString(threat_model) << "\",";
    os << "\"perf_class\":\"" << ToString(perf_class) << "\",";
    os << "\"recommended_max_db_size\":" << recommended_max_db_size << ",";
    os << "\"backends\":[";
    first = true;
    for (auto b : backends) {
      if (!first) os << ",";
      os << "\"" << ToString(b) << "\"";
      first = false;
    }
    os << "],";
    os << "\"typical_query_comm_bytes\":" << typical_query_comm_bytes << ",";
    os << "\"typical_hint_size_bytes\":" << typical_hint_size_bytes;
    os << "}";
    return os.str();
  }
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_CAPABILITIES_H_
