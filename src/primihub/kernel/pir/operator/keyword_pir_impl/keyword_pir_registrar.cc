/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */

// Registers the APSI-based labeled-PSI keyword PIR algorithm under the name
// "apsi" so PirSelector / pir_inspect can see it and so the factory shim
// routes legacy PirType::KEY_PIR (mapped in common.h LegacyNameFor) here.
//
// APSI has two role-specific implementations (KeywordPirOperatorClient /
// KeywordPirOperatorServer). The registry expects a single Creator, so a
// thin facade dispatches on Options.role at OnExecute time. This file is
// only built when the build is configured with --define=microsoft-apsi=true
// (see keyword_pir_impl/BUILD); when APSI is disabled, the algorithm simply
// does not appear in the registry and selector skips it.

#include <memory>

#include <glog/logging.h>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/common.h"
#include "src/primihub/kernel/pir/operator/base_pir.h"
#include "src/primihub/kernel/pir/operator/capabilities.h"
#include "src/primihub/kernel/pir/operator/registry.h"
#include "src/primihub/kernel/pir/operator/keyword_pir_impl/keyword_pir_client.h"
#include "src/primihub/kernel/pir/operator/keyword_pir_impl/keyword_pir_server.h"

namespace primihub::pir {

class KeywordPirOperator : public BasePirOperator {
 public:
  explicit KeywordPirOperator(const Options& options)
      : BasePirOperator(options) {}

  retcode OnExecute(const PirDataType& input, PirDataType* result) override {
    if (result == nullptr) {
      LOG(ERROR) << "KeywordPirOperator: result pointer is null";
      return retcode::FAIL;
    }
    if (RoleValidation::IsClient(options_.role)) {
      KeywordPirOperatorClient client(options_);
      return client.Execute(input, result);
    }
    if (RoleValidation::IsServer(options_.role)) {
      KeywordPirOperatorServer server(options_);
      return server.Execute(input, result);
    }
    LOG(ERROR) << "KeywordPirOperator: unsupported role "
               << static_cast<int>(options_.role);
    return retcode::FAIL;
  }
};

namespace {

// Capability profile for APSI (Chen et al. — Microsoft labeled PSI).
// Numbers reflect the upstream paper + APSI default param presets. Where
// the literature reports ranges we pick the conservative value so Selector
// does not over-prefer APSI for huge databases it would not actually serve
// in production.
PirCapabilities KeywordPirCaps() {
  PirCapabilities caps;
  caps.query_types = {QueryType::Keyword};
  caps.min_servers = 1;
  caps.max_servers = 1;
  // APSI builds a SenderDb in an offline phase (BinBundle / OPRF table).
  // The hint is per-database — change the keyword set, regenerate.
  caps.needs_preprocess = true;
  caps.hint_per_database = true;
  caps.threat_model = ThreatModel::SemiHonest;
  // Online phase reports ~seconds at million-scale sender DBs on CPU.
  caps.perf_class = PerfClass::Seconds;
  // Practical upper bound from APSI's own benchmarks; beyond ~1e8 keys
  // memory + preprocessing become operationally unfriendly.
  caps.recommended_max_db_size = 100'000'000ULL;  // 1e8
  caps.backends = {Backend::CPU};
  // Query-side ciphertexts at default 4K poly modulus run ~200 KB; the
  // exact figure depends on PSIParams chosen at session setup. Use a
  // realistic mid-range default so Selector ranks against comm-light
  // index PIRs sensibly.
  caps.typical_query_comm_bytes = 200 * 1024;
  // SenderDb size varies widely (10 MB - 1 GB depending on key count and
  // label width); leave 0 because hint_per_database = true makes that
  // signal lifecycle-managed rather than a flat number.
  caps.typical_hint_size_bytes = 0;
  caps.is_real = true;  // real Microsoft APSI integration
  return caps;
}

PirRegistrar<KeywordPirOperator> apsi_registrar_("apsi", KeywordPirCaps());

}  // namespace

}  // namespace primihub::pir
