/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/double_pir.h"

#include <string>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/double_pir/double_pir_runtime.h"
#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode DoublePirOperator::OnExecute(const PirDataType& /*input*/,
                                     PirDataType* /*result*/) {
  // Phase 4 scaffolding milestone — the operator now drives the
  // DoublePirRuntime smoke path so we have a definitive signal that
  // the @simplepir//:simplepir_c_kernels cc_library is linked. The
  // real DoublePIR query path (LWE secret + hint + answer + decode)
  // lands in a follow-up to task 5.5 once the Go algorithmic layer is
  // ported into C++.
  if (!double_pir::kDoublePirRuntimeVendored) {
    LOG(ERROR)
        << "DoublePirOperator: runtime not vendored — build with "
        << "--define=enable_double_pir_real=1 and provide @simplepir "
        << "bazel override (see openspec/changes/primihub-pir-multi-algo "
        << "design.md Phase 4).";
    return retcode::FAIL;
  }

  std::string err;
  auto rc = double_pir::DoublePirRuntime::Instance().SmokeMatMulVec(&err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "DoublePirOperator: runtime smoke failed: " << err;
    return retcode::FAIL;
  }

  // Honest skeleton signal: the kernel link works and matmul is correct,
  // but the algorithmic core has not landed yet. Returning FAIL keeps
  // callers from assuming a real PIR query happened.
  LOG(WARNING)
      << "DoublePirOperator: runtime smoke PASS but full DoublePIR "
      << "query path not yet implemented; returning FAIL until task 5.5 "
      << "lands the LWE protocol port.";
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
