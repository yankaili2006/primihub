/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir.h"

#include <string>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/simple_pir/simple_pir_runtime.h"

namespace primihub::pir {

retcode SimplePirOperator::OnExecute(const PirDataType&, PirDataType*) {
  if (!simple_pir::kSimplePirRuntimeVendored) {
    LOG(ERROR)
        << "SimplePirOperator: runtime not vendored — build with "
        << "--define=enable_simple_pir_real=1 and provide @simplepir "
        << "bazel override (see openspec task 7.2 Phase 6).";
    return retcode::FAIL;
  }
  std::string err;
  auto rc = simple_pir::SimplePirRuntime::Instance().SmokeMatMul(&err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SimplePirOperator: runtime smoke failed: " << err;
    return retcode::FAIL;
  }
  LOG(WARNING)
      << "SimplePirOperator: runtime smoke PASS but full SimplePIR "
      << "query path not yet implemented; returning FAIL until task 7.2 "
      << "lands the LWE port.";
  return retcode::FAIL;
}

namespace {

PirCapabilities SimpleCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Index};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;       // public matrix A is pre-shared
  caps.hint_per_database = true;      // H = A * D depends on D
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;  // ~300 ms at 1 GB per paper §6
  caps.recommended_max_db_size = 100'000'000ULL;  // 1e8 sweet spot
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 121 * 1024;  // ~121 KB (paper Table 2)
  caps.typical_hint_size_bytes = 121ULL * 1024 * 1024;  // ~121 MB at 1 GB
  return caps;
}

PirRegistrar<SimplePirOperator> simple_pir_registrar_("simple_pir", SimpleCaps());

}  // namespace

}  // namespace primihub::pir
