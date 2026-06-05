/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_pir.h"

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode FrodoPirOperator::OnExecute(const PirDataType&, PirDataType*) {
  LOG(ERROR) << "FrodoPirOperator: skeleton only — wrapper for "
                "brave/frodo-pir pending (openspec).";
  return retcode::FAIL;
}

namespace {

PirCapabilities FrodoCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  // Paper §5: ~30 ms server-side at 1 GB; classify as Ms.
  caps.perf_class = PerfClass::Ms;
  caps.recommended_max_db_size = 100'000'000ULL;
  caps.backends = {Backend::CPU};
  // FrodoPIR's online traffic is smaller than SimplePIR (paper Table 1).
  caps.typical_query_comm_bytes = 64 * 1024;
  caps.typical_hint_size_bytes = 40ULL * 1024 * 1024;  // ~40 MB at 1 GB
  return caps;
}

PirRegistrar<FrodoPirOperator> frodo_pir_registrar_("frodo_pir", FrodoCaps());

}  // namespace

}  // namespace primihub::pir
