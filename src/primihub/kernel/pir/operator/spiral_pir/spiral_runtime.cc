/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Two compile modes selected by PIR_SPIRAL_RUNTIME_VENDORED:
 *
 *   * Defined (set by BUILD when --define=enable_spiral_real=1): pulls in
 *     upstream "spiral.h" and implements EnsureInitialized against the
 *     real setup_constants / generate_gadgets / build_table. The three
 *     cryptographic methods (ClientEncode, ServerProcess, ClientDecode)
 *     are clear TODOs returning retcode::FAIL with pointer to the spec —
 *     wiring the full query path is a multi-day cryptographic engineering
 *     task that should land in its own commit.
 *
 *   * Undefined (default): every method returns retcode::FAIL with a
 *     "spiral runtime not vendored" message. The header still works for
 *     callers, kSpiralRuntimeVendored is false, and the skeleton tests
 *     continue to pass.
 *
 * Either way, the std::mutex + singleton + params-locking semantics are
 * identical and unit-testable.
 */
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_runtime.h"

#include <atomic>
#include <cstring>
#include <mutex>

#include <glog/logging.h>

#ifdef PIR_SPIRAL_RUNTIME_VENDORED
// These globals are declared at namespace scope in upstream spiral.h and
// mutated by setup_constants / load_db / generate_setup_and_query /
// process_crtd_query. Including the header in this single TU keeps the
// namespace pollution contained.
#include "spiral.h"
// Upstream defines these in spiral.cpp at namespace scope but never declares
// them in spiral.h, so forward-declare for our use.
void setup_constants();
void testHighRate(size_t, size_t, size_t) {}
void generate_gadgets();
void build_table();
void generate_setup_and_query(size_t idx,
                              uint64_t** g_C_fft_crtd,
                              uint64_t** g_Q_crtd,
                              uint64_t** g_Ws_fft,
                              bool encodeCompressedSetupData);
void load_db();
void do_test();
// Upstream globals consumed by load_db / do_test, defined at namespace
// scope in spiral.cpp L16-1013. Forward-declared here so SmokeTest can
// configure the random_data branch of load_db without main()-style args.
extern bool random_data;
extern bool has_file;
extern bool load;
extern bool checking_for_debug;
extern bool show_diff;
extern unsigned long dummyWorkingSet;
extern unsigned long max_trials;
#endif

namespace primihub::pir::spiral {

#ifdef PIR_SPIRAL_RUNTIME_VENDORED
const bool kSpiralRuntimeVendored = true;
#else
const bool kSpiralRuntimeVendored = false;
#endif

struct SpiralRuntime::Impl {
  std::mutex mu;
  std::atomic<bool> initialized{false};
  SpiralParams locked{0, 0, 0};
};

SpiralRuntime& SpiralRuntime::Instance() {
  static SpiralRuntime kSingleton;
  return kSingleton;
}

SpiralRuntime::SpiralRuntime() : impl_(new Impl()) {}
SpiralRuntime::~SpiralRuntime() { delete impl_; }

SpiralParams SpiralRuntime::locked_params() const {
  std::lock_guard<std::mutex> g(impl_->mu);
  return impl_->locked;
}

retcode SpiralRuntime::EnsureInitialized(const SpiralParams& p,
                                         std::string* err) {
  if (err == nullptr) return retcode::FAIL;
  std::lock_guard<std::mutex> g(impl_->mu);

  if (impl_->initialized.load()) {
    if (impl_->locked.nu_1 != p.nu_1 || impl_->locked.nu_2 != p.nu_2) {
      *err =
          "SpiralRuntime: already initialized with different params "
          "(single-params-per-process limitation — see "
          "docs/pir/multi-algo-guide.md \"Spiral lifecycle\" section). "
          "Locked: nu_1=" +
          std::to_string(impl_->locked.nu_1) + " nu_2=" +
          std::to_string(impl_->locked.nu_2) + "; requested: nu_1=" +
          std::to_string(p.nu_1) + " nu_2=" + std::to_string(p.nu_2);
      return retcode::FAIL;
    }
    return retcode::SUCCESS;  // idempotent
  }

#ifdef PIR_SPIRAL_RUNTIME_VENDORED
  // Upstream's main() at spiral.cpp L1229-1320 does exactly this sequence
  // before any query work. We mirror it but suppress the cout chatter
  // upstream emits — the HF==0 invariant check inside setup_constants
  // will still abort the process on violation, which is the documented
  // upstream behavior we intentionally do not paper over.
  ::omp_set_num_threads(1);
  ::build_table();
  ::scratch = reinterpret_cast<uint64_t*>(
      std::malloc(::crt_count * ::poly_len * sizeof(uint64_t)));
  if (::scratch == nullptr) {
    *err = "SpiralRuntime: scratch malloc failed";
    return retcode::FAIL;
  }
  ::ntt_qprime = new intel::hexl::NTT(2048, ::arb_qprime);
  ::num_expansions = static_cast<long>(p.nu_1);
  ::further_dims = static_cast<long>(p.nu_2);
  ::total_n = (1ULL << p.nu_1) * (1ULL << p.nu_2);
  ::IDX_TARGET = 0;  // overwritten per query
  ::IDX_DIM0 = 0;
  ::setup_constants();
  ::generate_gadgets();

  impl_->locked = p;
  impl_->initialized.store(true);
  return retcode::SUCCESS;
#else
  (void)p;
  *err =
      "SpiralRuntime: not vendored. Build with --define=enable_spiral_real=1 "
      "and provide @hexl + @spiral_pir bazel overrides (see "
      "openspec/changes/primihub-pir-multi-algo design.md §D7 \"Build "
      "wiring\"). Until then ClientEncode/ServerProcess/ClientDecode "
      "return FAIL.";
  return retcode::FAIL;
#endif
}

retcode SpiralRuntime::ClientEncode(uint64_t /*index*/,
                                    std::vector<uint8_t>* /*wire_blob*/,
                                    std::string* err) {
  if (err == nullptr) return retcode::FAIL;
  // Architectural blocker (verified empirically by attempt 21a73ad6
  // followup): upstream menonsamir/spiral cannot be cleanly split into
  // client-encode + server-process roles. Specifically:
  //
  //   * src/spiral.cpp L1546: generate_setup runs keygen, writing secret
  //     key into globals S_mp/Sp_mp/sr_mp. It does NOT write the public
  //     setup data (g_Ws_fft) unless direct_upload=true. With the wiki
  //     config we picked (direct_upload=false, query_size=14KB), g_Ws_fft
  //     stays nullptr after generate_setup_and_query returns.
  //
  //   * src/spiral.cpp L2046: runConversionImproved (called from
  //     process_crtd_query) reads IDX_TARGET from globals AND constructs
  //     the public encryptions (W_exp_v / W_exp_right_v) via
  //     getPublicEncryptions, which itself uses S_mp/Sp_mp (the secret
  //     key). Upstream is a single-process benchmark, not a true
  //     client/server PIR.
  //
  // True client-server split therefore requires upstream refactor:
  // (a) move getPublicEncryptions out of runConversionImproved so the
  // client can call it with the secret key and ship g_Ws_fft to the
  // server; (b) thread IDX_TARGET through runConversionImproved as a
  // parameter rather than a global. That's multi-week deep-crypto work.
  //
  // For v1 the SpiralPirOperator will either (i) run client+server roles
  // in the same OnExecute (no real PIR security, but functional smoke
  // test) or (ii) bundle the secret key with the query (defeats PIR,
  // explicit toy mode). Both are documented as v1 limitations.
#ifdef PIR_SPIRAL_RUNTIME_VENDORED
  *err =
      "SpiralRuntime::ClientEncode requires upstream refactor — "
      "menonsamir/spiral is a single-process benchmark; generate_setup "
      "does not populate g_Ws_fft (verified). See spiral_runtime.cc "
      "comment block for refactor scope.";
#else
  *err =
      "SpiralRuntime::ClientEncode not vendored. Build with "
      "--define=enable_spiral_real=1 (see EnsureInitialized error).";
#endif
  return retcode::FAIL;
}

retcode SpiralRuntime::ServerProcess(
    const std::vector<uint8_t>& /*wire_blob*/,
    const std::vector<std::string>& /*records*/,
    std::vector<uint8_t>* /*wire_response*/, std::string* err) {
  if (err == nullptr) return retcode::FAIL;
  // TODO(task 4.4 followup): implement against upstream
  //   parse wire_blob into (g_C_fft_crtd, g_Q_crtd, g_Ws_fft)
  //   populate the global `B` buffer from records (or write a tmpfile + load_db)
  //   allocate ExpansionLocals / FurtherDimsLocals (see spiral.h L96-115)
  //   process_crtd_query(expansionLocals, furtherDimsLocals,
  //                      g_C_fft_crtd, g_Q_crtd, g_Ws_fft);
  //   modswitch(furtherDimsLocals.result, furtherDimsLocals.cts);
  //   memcpy furtherDimsLocals.result into wire_response
  *err =
      "SpiralRuntime::ServerProcess not yet implemented — see "
      "openspec/changes/primihub-pir-multi-algo task 4.4 followup. "
      "Upstream API: spiral.cpp L2337 process_crtd_query + L2459 modswitch.";
  return retcode::FAIL;
}

retcode SpiralRuntime::ClientDecode(
    const std::vector<uint8_t>& /*wire_response*/, std::string* /*out_value*/,
    std::string* err) {
  if (err == nullptr) return retcode::FAIL;
  // TODO(task 4.4 followup): implement against extracted decode logic
  //   from check_final (spiral.cpp L2200ish — note check_final couples
  //   recovery + verification, must split for application use). The
  //   secret key (S_mp / Sp_mp / sr_mp) retained from ClientEncode is
  //   needed here.
  *err =
      "SpiralRuntime::ClientDecode not yet implemented — see "
      "openspec/changes/primihub-pir-multi-algo task 4.4 followup. "
      "Need to factor recover-path out of upstream check_final.";
  return retcode::FAIL;
}


retcode SpiralRuntime::SmokeTest(uint64_t index, std::string* err) {
  if (err == nullptr) return retcode::FAIL;
  std::lock_guard<std::mutex> g(impl_->mu);
  if (!impl_->initialized.load()) {
    *err = "SpiralRuntime::SmokeTest requires EnsureInitialized first";
    return retcode::FAIL;
  }
#ifdef PIR_SPIRAL_RUNTIME_VENDORED
  if (index >= impl_->locked.total_n) {
    *err = "SpiralRuntime::SmokeTest index " + std::to_string(index) +
           " out of range; total_n=" + std::to_string(impl_->locked.total_n);
    return retcode::FAIL;
  }
  // Same-process simulation. Configure upstream globals as if main() had
  // been called with `nu_1 nu_2 index a --random-data`:
  //   * IDX_TARGET / IDX_DIM0 select the query index
  //   * random_data=true makes load_db skip the file-based TODO branch
  //   * has_file/load=false keep load_db on the random_data path
  //   * checking_for_debug=true makes do_test invoke check_final
  ::IDX_TARGET = static_cast<long>(index);
  ::IDX_DIM0 = static_cast<long>(index / (1ULL << impl_->locked.nu_2));
  ::random_data = true;
  ::has_file = false;
  ::load = false;
  ::checking_for_debug = true;
  ::show_diff = false;
  const unsigned long total_n_val = impl_->locked.total_n;
  unsigned long ws = (1UL << 25) / (total_n_val ? total_n_val : 1);
  if (ws > static_cast<unsigned long>(::poly_len)) ws = ::poly_len;
  if (ws == 0) ws = 1;
  ::dummyWorkingSet = ws;
  ::max_trials = 1;

  ::load_db();
  ::do_test();
  return retcode::SUCCESS;
#else
  (void)index;
  *err =
      "SpiralRuntime::SmokeTest not vendored. Build with "
      "--define=enable_spiral_real=1 (see EnsureInitialized error).";
  return retcode::FAIL;
#endif
}

}  // namespace primihub::pir::spiral
