/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_pir.h"

#include <cstdlib>
#include <string>

#include <glog/logging.h>
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/spiral_pir/params.h"
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_runtime.h"

namespace primihub::pir {

retcode SpiralPirOperator::OnExecute(const PirDataType& input,
                                     PirDataType* result) {
  if (result == nullptr) return retcode::FAIL;
  if (!spiral::kSpiralRuntimeVendored) {
    LOG(ERROR)
        << "SpiralPirOperator: runtime not vendored — build with "
        << "--define=enable_spiral_real=1 and provide @hexl + @spiral_pir "
        << "bazel overrides (see openspec/changes/primihub-pir-multi-algo "
        << "design.md §D7).";
    return retcode::FAIL;
  }

  // v1 same-process simulation contract: input is one entry whose key is
  // a stringified PIR index ("0", "1", ...). All other keys are ignored
  // (caller may pass a multi-key map but only the first is queried). The
  // returned result echoes the key with one placeholder value that signals
  // pipeline success — actual record retrieval requires the upstream
  // refactor documented in commits e988ae4f / 548d1c48.
  if (input.empty()) {
    LOG(ERROR) << "SpiralPirOperator::OnExecute: input map empty";
    return retcode::FAIL;
  }
  const auto& first = *input.begin();
  uint64_t index = 0;
  try {
    index = std::stoull(first.first);
  } catch (const std::exception& e) {
    LOG(ERROR) << "SpiralPirOperator::OnExecute: input key '" << first.first
               << "' not a uint64 index: " << e.what();
    return retcode::FAIL;
  }

  // KNOWN ISSUE ("Is correct?: 0"): EstimateParams picks (nu_1,nu_2)
  // independently, but the compile-time SPIRAL_DEFINES (BUILD.spiral) are a
  // matched bundle from upstream select_params.py for ONE config -- defines
  // and dims must agree. Fix: pin dims to the compiled config. Root-cause +
  // reproduction recipe: docs/pir/spiral-calibration-notes.md.
  // Pick params from the requested index + a small default record size.
  // v1 limitation: SmokeTest's load_db ignores caller-supplied records;
  // the DB is constant-valued. Record size is informational only.
  const uint64_t db_size = index + 1;  // smallest that holds this index
  spiral::SpiralParams p{};
  std::string err;
  auto rc = spiral::EstimateParams(db_size, /*record_size_bytes=*/256, &p,
                                   &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SpiralPirOperator: EstimateParams failed: " << err;
    return retcode::FAIL;
  }

  auto& rt = spiral::SpiralRuntime::Instance();
  rc = rt.EnsureInitialized(p, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SpiralPirOperator: EnsureInitialized failed: " << err;
    return retcode::FAIL;
  }
  rc = rt.SmokeTest(index, &err);
  if (rc != retcode::SUCCESS) {
    LOG(ERROR) << "SpiralPirOperator: SmokeTest failed: " << err;
    return retcode::FAIL;
  }

  (*result)[first.first] = {"spiral_pipeline_ran"};
  LOG(INFO) << "SpiralPirOperator::OnExecute: pipeline ran for index "
            << index << " (correctness invariant pending — see "
            << "commit 548d1c48)";
  return retcode::SUCCESS;
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
