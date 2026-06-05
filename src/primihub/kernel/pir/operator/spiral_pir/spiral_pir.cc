/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_pir.h"

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode SpiralPirOperator::OnExecute(const PirDataType& /*input*/,
                                     PirDataType* /*result*/) {
  LOG(ERROR)
      << "SpiralPirOperator: skeleton only — real query path not yet wired. "
      << "Vendor menonsamir/spiral via thirdparty/pir/BUILD.spiral and "
      << "replace this method (see openspec/changes/primihub-pir-multi-algo "
      << "tasks 4.1-4.7).";
  return retcode::FAIL;
}

namespace {

PirCapabilities SpiralCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = false;
  caps.hint_per_database = false;
  caps.threat_model = ThreatModel::SemiHonest;
  // Paper reports ~2-3 s server-side compute at 1e8 rows; classify as
  // Seconds. Below ~1e6 the algorithm crosses into sub-second territory,
  // but Selector ranking already accounts for db_size, so the conservative
  // (Seconds) class here doesn't penalize medium DBs in practice.
  caps.perf_class = PerfClass::Seconds;
  caps.recommended_max_db_size = 1'000'000'000ULL;  // 1e9
  caps.backends = {Backend::CPU};                   // AVX2/CUDA in P7
  // Per-query online traffic per Menon & Wu §6 (Table 1) — single round
  // ~14 KB ciphertext + ~12 KB extraction. These numbers help Selector
  // rank against communication-sensitive choices like YPIR.
  caps.typical_query_comm_bytes = 14 * 1024 + 12 * 1024;
  caps.typical_hint_size_bytes = 0;
  return caps;
}

PirRegistrar<SpiralPirOperator> spiral_pir_registrar_("spiral", SpiralCaps());

}  // namespace

}  // namespace primihub::pir
