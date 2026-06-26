/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_pir.h"

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/registry.h"

namespace primihub::pir {

retcode TiptoePirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  (void)input;
  (void)result;
  // SKELETON (task 1.1a): the registered algorithm + capability profile +
  // dependency edge are in place, but the real BFV-on-SimplePIR query path is
  // not yet vendored/ported. Lands incrementally in chunks 1.1b-1.1f (rlwe
  // vendor -> params/secret -> client -> server/hint -> OnExecute wiring).
  // See docs/pir/tiptoe-port-plan.md.
  LOG(ERROR) << "TiptoePirOperator: not yet implemented (skeleton). The "
                "BFV-on-SimplePIR query path lands in primihub-pir-cuda-tiptoe "
                "chunks 1.1b-1.1f.";
  return retcode::FAIL;
}

namespace {

PirCapabilities TiptoeCaps() {
  PirCapabilities caps;
  caps.is_real = false;  // skeleton — flips true when OnExecute is wired (1.1f)
  caps.query_types = {QueryType::Semantic};
  caps.min_servers = 1;
  caps.max_servers = 1;
  caps.needs_preprocess = true;   // offline per-database hint
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  caps.perf_class = PerfClass::SubSecond;
  caps.recommended_max_db_size = 100'000'000ULL;  // 1e8 (paper scale)
  caps.backends = {Backend::CPU};
  caps.typical_query_comm_bytes = 16'384;          // ~16 KB BFV query
  caps.typical_hint_size_bytes = 2'000'000'000ULL;  // ~2 GB hint at scale
  return caps;
}

PirRegistrar<TiptoePirOperator> tiptoe_registrar_("tiptoe", TiptoeCaps());

}  // namespace

}  // namespace primihub::pir
