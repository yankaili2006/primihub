/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::double_pir::hint_link — LinkContext-aware transport
 * helpers for shipping a DoublePirHint between non-colluding DoublePIR
 * peers.
 *
 * Task 5.6 chunk 6: in production deployment the two DoublePIR servers
 * are non-colluding; each can derive the same DoublePirHint locally
 * because hint generation is deterministic given (DB, seed). To save
 * compute, one peer ("primary") runs HintGen::Compute and broadcasts
 * the serialized result over MultiPeerPirOperator's existing
 * Send/Recv wire path, and the other peer ("secondary") installs the
 * received hint instead of re-running the O(L·M·n) Setup.
 *
 * This file is the *transport layer only*. Operator-level wiring
 * (role detection, OnExecute integration, HintCache install on
 * receive) lands in a follow-up chunk so the existing single-process
 * test surface and benchmarks stay intact.
 *
 * Both helpers are pure functions over an injected LinkContext*;
 * they do not touch HintCache. Callers decide whether to Put() the
 * received hint into the LRU after computing the
 * (l, m, p, logq, db_fingerprint) key locally.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_LINK_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_LINK_H_

#include <string>
#include <vector>

#include "src/primihub/common/common.h"
#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"
#include "src/primihub/util/network/link_context.h"

namespace primihub::pir::double_pir {

// Default key used for cross-peer hint shipping. Callers may override
// to namespace multiple concurrent hints (e.g., versioned A/B rollout,
// or one hint per (l, m, p, logq) shape if a node serves several).
inline constexpr const char* kDefaultHintWireKey = "double_pir.hint.v1";

// BroadcastHint serializes `hint` via SerializeHint (PHHB v1) and Sends
// a copy to every node in `peers` under `key`.
//
// Semantics:
//   * `peers` empty (regardless of link state) — returns SUCCESS without
//     any I/O. Lets callers wire BroadcastHint unconditionally; the
//     single-process path stays a no-op.
//   * `link == nullptr` AND `peers` non-empty — FAIL (caller bug).
//   * Any Send returning non-SUCCESS short-circuits with that retcode;
//     `err` is populated with the failed peer index and its id.
//
// The serialized payload is identical across peers. DoublePIR's
// non-colluding assumption rules out per-peer differentiation at the
// transport layer — the privacy guarantee comes from the protocol
// (each server only sees its share of the *query*), not from hiding
// hint bytes between servers.
retcode BroadcastHint(network::LinkContext* link,
                      const std::vector<Node>& peers,
                      const DoublePirHint& hint,
                      const std::string& key = kDefaultHintWireKey,
                      std::string* err = nullptr);

// ReceiveHint Recvs a serialized hint from `peer` via `link` under
// `key`, then deserializes it into *hint_out.
//
// On any wire-level error (Recv failure, magic mismatch, version
// mismatch, truncated body, dimension overflow) returns retcode::FAIL
// with `err` populated. On success, *hint_out is fully populated and
// caller-owned.
//
// Note: ReceiveHint does NOT install into HintCache. Callers that want
// LRU-managed deduplication should Put() the result themselves after
// recomputing the (l, m, p, logq, db_fingerprint) key locally — the
// fingerprint depends on DB bytes the secondary already holds.
retcode ReceiveHint(network::LinkContext* link, const Node& peer,
                    DoublePirHint* hint_out,
                    const std::string& key = kDefaultHintWireKey,
                    std::string* err = nullptr);

}  // namespace primihub::pir::double_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_LINK_H_
