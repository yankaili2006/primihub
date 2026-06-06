/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SpiralRuntime — thin C++ facade over menonsamir/spiral upstream.
 *
 * Hides three properties of upstream that we do not want to leak into
 * SpiralPirOperator or any other primihub TU:
 *
 *   1. Globals at namespace scope (spiral.h L67-104). Upstream assumes a
 *      single (num_expansions, further_dims) tuple for the lifetime of the
 *      process; setup_constants() (spiral.cpp L127-187) writes the matrix
 *      tables once with no free path. SpiralRuntime is a process-singleton
 *      that gates EnsureInitialized so the first caller's params win and
 *      subsequent callers with different params get retcode::FAIL.
 *
 *   2. No thread safety. Upstream forces omp_set_num_threads(1) and uses
 *      no mutexes. Every facade call takes a process-wide std::mutex so
 *      concurrent OnExecute invocations (multiple operator instances, or
 *      multiple primihub tasks on the same node) are serialized.
 *
 *   3. Raw uint64_t* query/response arrays with no serialize helpers
 *      (poly.h L126-127 + poly.cpp L518-540 are commented-out stubs).
 *      The facade pins a length-prefixed wire format so SpiralPirOperator
 *      only ever passes std::vector<uint8_t> blobs over link_ctx_ref.
 *
 * Build wiring: this header is consumed by SpiralPirOperator at
 * compile time only — it forward-declares no upstream types. The .cc
 * either includes upstream's spiral.h (when @hexl + @spiral_pir are in
 * WORKSPACE) and forwards facade calls into real upstream code, OR
 * returns retcode::FAIL with a clear "spiral runtime not vendored"
 * message. Selection is via the kSpiralRuntimeVendored constexpr below.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_RUNTIME_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_RUNTIME_H_

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/spiral_pir/params.h"

namespace primihub::pir::spiral {

// True iff this build links against the real menonsamir/spiral library.
// When false, every facade method returns retcode::FAIL with a clear
// "not vendored" message — the operator is still constructible and the
// skeleton tests still pass.
extern const bool kSpiralRuntimeVendored;

// Process-wide singleton wrapping all access to upstream Spiral. Every
// public method is thread-safe (a single internal std::mutex serializes
// all calls). All inputs and outputs are POD / std::vector — no upstream
// types cross this interface.
class SpiralRuntime {
 public:
  // Singleton accessor. Constructed lazily on first use.
  static SpiralRuntime& Instance();

  // Initializes upstream globals with the given params. Idempotent on
  // identical params. Returns retcode::FAIL with a populated `err` if
  // (a) called a second time with different params (single-params-per-
  // process limitation; documented in docs/pir/multi-algo-guide.md), or
  // (b) kSpiralRuntimeVendored is false.
  retcode EnsureInitialized(const SpiralParams& p, std::string* err);

  // CLIENT role: generate a fresh secret key + query the given index.
  // Output `wire_blob` is a length-prefixed concatenation:
  //   [u32 keys_len][keys_bytes][u32 query_len][query_bytes]
  // The internal secret key is retained on the singleton and consumed
  // by ClientDecode below; callers must invoke ClientDecode exactly
  // once per ClientEncode in the same OnExecute scope.
  retcode ClientEncode(uint64_t index,
                       std::vector<uint8_t>* wire_blob,
                       std::string* err);

  // SERVER role: parse the client's wire blob, load the database from
  // `records` (record[i] is the value at PIR index i; vector size must
  // equal SpiralParams::total_n), run process_crtd_query, and emit the
  // modswitched response bytes into `wire_response`.
  retcode ServerProcess(const std::vector<uint8_t>& wire_blob,
                        const std::vector<std::string>& records,
                        std::vector<uint8_t>* wire_response,
                        std::string* err);

  // CLIENT role: decrypt the server's response with the secret key
  // retained from the matching ClientEncode call and write the recovered
  // record value into `out_value`.
  retcode ClientDecode(const std::vector<uint8_t>& wire_response,
                       std::string* out_value,
                       std::string* err);

  // Returns the params currently locked in by EnsureInitialized, or
  // {0, 0, 0} if Init has never succeeded.
  SpiralParams locked_params() const;

 private:
  SpiralRuntime();
  ~SpiralRuntime();
  SpiralRuntime(const SpiralRuntime&) = delete;
  SpiralRuntime& operator=(const SpiralRuntime&) = delete;

  // PImpl: hides the std::mutex / atomic / upstream-state pointers so
  // this header is compileable without @hexl / @spiral_pir in deps.
  struct Impl;
  Impl* impl_;
};

}  // namespace primihub::pir::spiral

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_SPIRAL_RUNTIME_H_
