/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir.h"

#include <string>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_runtime.h"

namespace primihub::pir {

retcode YpirOperator::OnExecute(const PirDataType&, PirDataType*) {
  if (!ypir::kYpirRuntimeVendored) {
    LOG(ERROR)
        << "YpirOperator: runtime not vendored — build with "
        << "--define=enable_ypir_real=1 and provide @ypir bazel "
        << "override (see openspec task 7.3 Phase 6).";
    return retcode::FAIL;
  }
  std::string err;
  auto rc = ypir::YpirRuntime::Instance().SmokeMatMulVecPacked(&err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "YpirOperator: runtime smoke failed: " << err;
    return retcode::FAIL;
  }
  LOG(WARNING)
      << "YpirOperator: runtime smoke PASS but full YPIR query path "
      << "not yet implemented; returning FAIL until task 7.3 lands the "
      << "Rust-to-C++ algorithmic port.";
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
