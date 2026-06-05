/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir.h"

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode YpirOperator::OnExecute(const PirDataType&, PirDataType*) {
  LOG(ERROR) << "YpirOperator: skeleton only (openspec follow-up).";
  return retcode::FAIL;
}

namespace {

PirCapabilities YpirCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;       // per-db hint trades for low traffic
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;  // paper: ~500 ms at 1 GB
  caps.recommended_max_db_size = 1'000'000'000ULL;  // 1e9
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 900;  // <1 KB — YPIR's distinguishing feature
  caps.typical_hint_size_bytes = 8ULL * 1024 * 1024;  // ~8 MB at 1 GB
  return caps;
}

PirRegistrar<YpirOperator> ypir_registrar_("ypir", YpirCaps());

}  // namespace

}  // namespace primihub::pir
