/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::simple_pir::SerializeHint / DeserializeHint —
 * wire-format helpers for shipping a SimplePirHint between processes
 * or persisting it to disk. SimplePIR sibling of DoublePIR's
 * hint_serialize (task 5.6 chunk 3, commit 3595f95f).
 *
 * Format (little-endian throughout):
 *
 *   magic   : "PSHB"     4 bytes  ("PrimiHub Simple Hint Blob")
 *   version : uint16     2 bytes  (current = 1)
 *   reserved: uint16     2 bytes  (must be 0)
 *   info    : 10 × uint64 80 bytes
 *             num, row_length, packing, ne, x, p, logq, basis, squishing, cols
 *   for each of 2 matrices (A, H):
 *     rows  : uint64       8 bytes
 *     cols  : uint64       8 bytes
 *     data  : rows*cols × uint32 (little-endian)
 *
 * Fixed overhead = 4 + 2 + 2 + 80 + 2 × 16 = 120 bytes.
 *
 * Distinct PSHB magic (vs DoublePIR's PHHB) prevents cross-algorithm
 * accidents — a SimplePIR cache file deserialized as a DoublePIR
 * hint would silently produce 3 zero-shaped tail matrices, which is
 * loud at query time but not loud at deserialize time. The magic
 * check catches the swap at parse.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_SERIALIZE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_SERIALIZE_H_

#include <cstdint>
#include <string>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/simple_pir/hint_gen.h"

namespace primihub::pir::simple_pir {

inline constexpr uint16_t kSimpleHintWireVersion = 1;
inline constexpr char kSimpleHintMagic[4] = {'P', 'S', 'H', 'B'};

retcode SerializeHint(const SimplePirHint& hint, std::string* blob_out,
                      std::string* err = nullptr);

retcode DeserializeHint(const std::string& blob, SimplePirHint* hint_out,
                        std::string* err = nullptr);

}  // namespace primihub::pir::simple_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_SERIALIZE_H_
