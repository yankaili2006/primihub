/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode DoublePirOperator::OnExecute(const PirDataType& /*input*/,
                                     PirDataType* /*result*/) {
  LOG(ERROR)
      << "DoublePirOperator: skeleton only — real query path not yet wired. "
      << "Vendor ahenzinger/simplepir via thirdparty/pir/BUILD.simplepir and "
      << "replace this method (see openspec/changes/primihub-pir-multi-algo "
      << "tasks 5.4-5.10).";
  return retcode::FAIL;
}

namespace {

PirCapabilities DoubleCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  // EXACTLY two servers — DoublePIR's privacy proof requires two
  // independently sampled secrets; using one server collapses to SimplePIR,
  // using three would need a separate three-party variant.
  caps.min_servers = 2;
  caps.max_servers = 2;
  // Per-database public hint of size O(sqrt(N) * lambda) is computed once
  // and shared with every client; clients cache it locally.
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  // The privacy guarantee REQUIRES the two servers not to collude. Selector
  // will only return this algorithm when Constraints.assume_non_colluding.
  caps.threat_model = ThreatModel::SemiHonestNonColluding;
  // Paper Table 3: ~12 ms server + ~5 ms client compute at 1 GB database.
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 1'000'000'000ULL;  // 1e9
  caps.backends = {Backend::CPU};                   // AVX2/CUDA in P7
  // Online traffic is the algorithm's main win: a single LWE ciphertext
  // pair (~4 KB at 1e8) — the large cost lives in the offline hint.
  caps.typical_query_comm_bytes = 4 * 1024;
  // Pre-shared hint at 1e8 / 1 GB database is on the order of 16 MB
  // (paper Table 3). Selector uses this when the client constraint
  // includes a memory budget for the cached hint.
  caps.typical_hint_size_bytes = 16ULL * 1024 * 1024;
  return caps;
}

PirRegistrar<DoublePirOperator> double_pir_registrar_("double_pir",
                                                      DoubleCaps());

}  // namespace

}  // namespace primihub::pir
