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
  // TODO(task 4.4 followup): implement against upstream
  //   generate_setup_and_query(idx, &g_C_fft_crtd, &g_Q_crtd, &g_Ws_fft, false);
  // serialize {keys_len, keys, query1_len, query1, query2_len, query2} into
  // wire_blob; retain S_mp / Sp_mp / sr_mp on the singleton for ClientDecode.
  *err =
      "SpiralRuntime::ClientEncode not yet implemented — see "
      "openspec/changes/primihub-pir-multi-algo task 4.4 followup. "
      "Upstream API: spiral.cpp L1540 generate_setup_and_query.";
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

}  // namespace primihub::pir::spiral
