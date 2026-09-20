/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIRANA_PIR_PIRANA_PIR_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIRANA_PIR_PIRANA_PIR_H_

#include "src/primihub/kernel/pir/operator/base_pir.h"

namespace primihub::pir {

// PIRANA — Faster Multi-query PIR via Constant-weight Codes
// (zju-abclab/PIRANA, eprint 2022/1401). Single-server BFV (SEAL) PIR where
// the query index is encoded as a constant-weight codeword; multi-query mode
// adds a Kuku cuckoo-hashed bundle step, with an optional compressed
// (communication-friendly) variant.
//
// Single-process model mirrors FrodoPirOperator: OnExecute drives the full
// protocol in one process (client gen_query → server gen_response → client
// extract_answer) over the vendored thirdparty/pir/pirana sources. Inter-party
// transport via LinkContext is N/A — caps.min_servers = caps.max_servers = 1.
//
// Input keys:
//   "db_content"    — vector of raw byte strings, one per payload (all must
//                     be <= payload_size bytes; short entries are zero-padded
//                     into payload_size/7 plaintext slots by the operator).
//   "query_indices" — vector of decimal index strings (one entry per query;
//                     a single index selects single-query PIR, multiple
//                     entries select batch PIR).
//   Optional tuning (decimal strings):
//   "payload_size"  — bytes per payload (default 256, max 273 for the
//                     default SEAL parms: 31 slots × (18-1 bits) / 8 = 65…
//                     actually slots hold 17-bit values, so 31*17/8 = 65
//                     bytes at poly 4096; larger payload_size raises the
//                     poly degree — see pir_parms.cc set_seal_parms).
//
// Output keys:
//   "recovered" — raw byte strings for each queried index, in query order.
class PiranaPirOperator : public BasePirOperator {
 public:
  explicit PiranaPirOperator(const Options& options)
      : BasePirOperator(options) {}
  ~PiranaPirOperator() override = default;
  retcode OnExecute(const PirDataType& input, PirDataType* result) override;
  static constexpr bool kIsSkeleton = false;
};

}  // namespace primihub::pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIRANA_PIR_PIRANA_PIR_H_
