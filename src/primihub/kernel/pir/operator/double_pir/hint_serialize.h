/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::double_pir::hint_serialize — wire-format helpers for
 * shipping a DoublePirHint between processes or persisting it to disk.
 *
 * Task 5.6 chunk 3 — sets up the LinkContext peer-split chunk (each
 * non-colluding server computes its hint slice locally then ships it
 * over MultiPeerPirOperator's SendToAllPeers). Also opens the door to
 * an out-of-process HintCache (persist the LRU on shutdown, reload at
 * boot).
 *
 * Format (little-endian throughout):
 *
 *   magic   : "PHHB"     4 bytes  ("PrimiHub Hint Blob")
 *   version : uint16     2 bytes  (current = 1)
 *   reserved: uint16     2 bytes  (must be 0)
 *   info    : 10 × uint64 80 bytes
 *             num, row_length, packing, ne, x, p, logq, basis, squishing, cols
 *   for each of 5 matrices (A1, A2, H1_squished, A2_copy_transposed, H2_msg):
 *     rows  : uint64       8 bytes
 *     cols  : uint64       8 bytes
 *     data  : rows*cols × uint32 (little-endian)
 *
 * Total fixed-overhead = 4 + 2 + 2 + 80 + 5 × 16 = 168 bytes.
 *
 * Non-goals: cryptographic integrity (caller wraps with HMAC if
 * needed), forward/backward compatibility (the format is versioned;
 * a future version bumps the constant and adds a parser branch).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_SERIALIZE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_SERIALIZE_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"

namespace primihub::pir::double_pir {

// Current on-the-wire version. Bump when the layout changes.
inline constexpr uint16_t kHintWireVersion = 1;

// 4-byte magic prefix. ASCII "PHHB".
inline constexpr char kHintMagic[4] = {'P', 'H', 'H', 'B'};

// Serializes hint into a self-contained byte string. blob_out must
// be non-null. Returns retcode::SUCCESS on success; the only failure
// mode in this revision is null pointer guard.
retcode SerializeHint(const DoublePirHint& hint, std::string* blob_out,
                      std::string* err = nullptr);

// Inverse of SerializeHint. Validates magic + version, then reads each
// matrix's [rows, cols, data]. On any framing error (short buffer,
// magic mismatch, version mismatch, dimension overflow), returns
// retcode::FAIL with err populated.
//
// Notes:
//   * The function does NOT validate the hint's mathematical
//     well-formedness — that requires the LWE params from the
//     producing process. It only validates wire-level invariants.
//   * On FAIL, *hint_out is left in an unspecified state.
retcode DeserializeHint(const std::string& blob, DoublePirHint* hint_out,
                        std::string* err = nullptr);

}  // namespace primihub::pir::double_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_SERIALIZE_H_
