/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SELECTOR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SELECTOR_H_

#include <cstdint>
#include <string>
#include <vector>
#include "src/primihub/kernel/pir/common.h"

namespace primihub::pir {

// Inputs to the algorithm selector. All fields are advisory; the selector
// excludes algorithms whose capabilities cannot satisfy the constraints.
struct Constraints {
  uint64_t db_size = 0;
  uint64_t element_size = 0;
  QueryType query_type = QueryType::Index;
  bool allow_two_server = false;
  bool client_can_cache_hint = false;
  bool assume_non_colluding = false;
  Backend preferred_backend = Backend::AUTO;
  LatencyBudget latency_budget = LatencyBudget::Any;
  ThreatModel min_threat_model = ThreatModel::SemiHonest;
  bool bandwidth_priority = false;
};

// Per-algorithm match info for the dry-run / capabilities CLI.
struct AlgoMatch {
  std::string algorithm;
  bool passes = false;
  uint64_t score = 0;  // higher is better
  uint64_t expected_latency_us_at_db_size = 0;
  uint64_t typical_query_comm_bytes = 0;
  std::vector<std::string> failed_checks;  // empty when passes=true
};

class PirSelector {
 public:
  // Returns algorithm names ranked best-first. Callers MAY iterate the list
  // for fallback. Returns an empty vector when no registered algorithm can
  // satisfy the constraints.
  std::vector<std::string> Recommend(const Constraints& c) const;

  // Same as Recommend but returns per-algorithm score + failed-check info,
  // intended for `primihub-cli pir auto --dry-run`.
  std::vector<AlgoMatch> RecommendWithRationale(const Constraints& c) const;
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SELECTOR_H_
