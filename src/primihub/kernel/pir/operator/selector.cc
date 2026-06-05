/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/selector.h"

#include <algorithm>
#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {
namespace {

// Lower numeric = faster perf class.
int PerfClassRank(PerfClass p) {
  switch (p) {
    case PerfClass::Ms: return 0;
    case PerfClass::SubSecond: return 1;
    case PerfClass::Seconds: return 2;
    case PerfClass::Tens: return 3;
  }
  return 99;
}

int LatencyBudgetRank(LatencyBudget b) {
  switch (b) {
    case LatencyBudget::Any: return 3;
    case LatencyBudget::Seconds: return 2;
    case LatencyBudget::SubSecond: return 1;
    case LatencyBudget::Ms: return 0;
  }
  return 99;
}

int ThreatModelRank(ThreatModel m) {
  switch (m) {
    case ThreatModel::Malicious: return 2;
    case ThreatModel::SemiHonestNonColluding: return 1;
    case ThreatModel::SemiHonest: return 0;
  }
  return 99;
}

// Evaluate one algorithm against the constraints. Populates m.failed_checks
// when not passing; sets m.score when passing.
void Evaluate(const std::string& algo, const PirCapabilities& caps,
              const Constraints& c, AlgoMatch* m) {
  m->algorithm = algo;
  m->typical_query_comm_bytes = caps.typical_query_comm_bytes;
  m->expected_latency_us_at_db_size = 0;  // populated below if passes

  // Hard filters
  if (!caps.query_types.count(c.query_type)) {
    m->failed_checks.push_back("query_type unsupported");
  }
  if (caps.min_servers >= 2 && !c.allow_two_server) {
    m->failed_checks.push_back("requires multi-server but allow_two_server=false");
  }
  if (caps.threat_model == ThreatModel::SemiHonestNonColluding &&
      !c.assume_non_colluding) {
    m->failed_checks.push_back("non-colluding assumption not declared");
  }
  if (!c.client_can_cache_hint && caps.typical_hint_size_bytes > 0 &&
      !caps.hint_per_database) {
    m->failed_checks.push_back("hint required but client cannot cache");
  }
  if (ThreatModelRank(caps.threat_model) < ThreatModelRank(c.min_threat_model)) {
    m->failed_checks.push_back("weaker threat model than required");
  }
  if (LatencyBudgetRank(c.latency_budget) < PerfClassRank(caps.perf_class)) {
    m->failed_checks.push_back("perf_class exceeds latency_budget");
  }
  if (c.db_size > 0 && caps.recommended_max_db_size > 0 &&
      c.db_size > caps.recommended_max_db_size * 10) {
    // We treat anything over 10× recommended max as out of scope, while
    // staying in tolerance for slight overshoots.
    m->failed_checks.push_back("db_size far exceeds recommended_max_db_size");
  }
  if (c.preferred_backend != Backend::AUTO &&
      !caps.backends.count(c.preferred_backend)) {
    // Soft check: caller's preferred backend isn't supported by this algo.
    // We don't reject — SelectBackend will fall back — but the score is
    // penalized so an algo with native support ranks higher.
    m->failed_checks.push_back("preferred backend not natively supported");
    // Don't return; allow scoring.
  }
  m->passes = m->failed_checks.empty() ||
              (m->failed_checks.size() == 1 &&
               m->failed_checks[0] == "preferred backend not natively supported");

  if (!m->passes) return;

  // Score: smaller perf_class + closer to recommended_max wins; bandwidth
  // priority flips comm bytes into a primary axis.
  uint64_t score = 1'000'000ull;
  score -= static_cast<uint64_t>(PerfClassRank(caps.perf_class)) * 100'000ull;
  if (c.bandwidth_priority) {
    score -= std::min<uint64_t>(caps.typical_query_comm_bytes / 64, 50'000);
  } else {
    score -= std::min<uint64_t>(caps.typical_query_comm_bytes / 1024, 5'000);
  }
  // Reward better threat model
  score += static_cast<uint64_t>(ThreatModelRank(caps.threat_model)) * 1'000;
  // Penalize big hint when client is the one caching
  if (caps.typical_hint_size_bytes > 0 && !caps.hint_per_database) {
    score -= std::min<uint64_t>(caps.typical_hint_size_bytes / (1ull << 20), 50'000);
  }
  // Soft penalty for backend mismatch
  if (c.preferred_backend != Backend::AUTO &&
      !caps.backends.count(c.preferred_backend)) {
    score -= 20'000;
  }
  m->score = score;
}

}  // namespace

std::vector<std::string> PirSelector::Recommend(const Constraints& c) const {
  auto matches = RecommendWithRationale(c);
  std::vector<std::string> names;
  for (const auto& m : matches) {
    if (m.passes) names.push_back(m.algorithm);
  }
  if (names.empty()) {
    LOG(INFO) << "PirSelector: no algorithm satisfies constraints";
  }
  return names;
}

std::vector<AlgoMatch> PirSelector::RecommendWithRationale(
    const Constraints& c) const {
  std::vector<AlgoMatch> matches;
  auto algos = PirRegistry::Instance().ListAlgorithms();
  for (const auto& algo : algos) {
    const auto* caps = PirRegistry::Instance().GetCapabilities(algo);
    if (!caps) continue;
    AlgoMatch m;
    Evaluate(algo, *caps, c, &m);
    matches.push_back(std::move(m));
  }
  // Sort: passing first (by score desc), then failing (by score desc)
  std::sort(matches.begin(), matches.end(),
            [](const AlgoMatch& a, const AlgoMatch& b) {
              if (a.passes != b.passes) return a.passes;
              return a.score > b.score;
            });
  return matches;
}

}  // namespace primihub::pir
