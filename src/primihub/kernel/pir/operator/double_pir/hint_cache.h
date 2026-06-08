/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::double_pir::HintCache — process-local LRU cache for
 * DoublePirHint, keyed by (l, m, p, logq, db_fingerprint).
 *
 * Task 5.6 chunk 2 follow-up to the HintGen refactor (chunk 1 /
 * commit 7303a83e). Repeat callers with the same (db, params) pair
 * now skip the O(L·M·n) Setup phase, which was the dominant cost
 * at every measured N (43 ms at N=64; 4510 ms at N=4M per the
 * baseline in docs/pir/benchmark.md).
 *
 * Wire-protocol note: the cached hint covers A1 / A2 / H1_squished
 * / A2_copy_transposed / H2_msg + info_after_setup. The squished-DB
 * itself is NOT cached — on a cache hit the caller re-runs the
 * cheap pair (db.ScalarAdd(p/2); db.Squish(10, 3)) which together
 * cost O(L·M), << O(L·M·n) Setup. This keeps cache memory bounded
 * to the hint matrices.
 *
 * Thread safety: instance methods take a std::mutex internally. The
 * singleton Instance() is initialized via Meyers-style local static.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_CACHE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_CACHE_H_

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "src/primihub/kernel/pir/operator/double_pir/hint_gen.h"

namespace primihub::pir::double_pir {

// Fingerprint computed from raw DB bytes + (l, m, p, logq). Cheap
// FNV-1a 64-bit so cache lookups stay sub-millisecond for the DB
// sizes we'll plausibly hit.
uint64_t FingerprintDb(const core::Database& db, const core::LweParams& params);

class HintCache {
 public:
  // Default max number of distinct hints retained. ~16 covers the
  // typical "single OnExecute" + "warmup + production" patterns
  // without growing the process RSS unboundedly. Each hint is on
  // the order of (sqrt(N) · n + n · delta · x · n + ...) uint32s —
  // 16 MB at N=1e8 per the paper's Table 3.
  static constexpr std::size_t kDefaultCapacity = 16;

  static HintCache& Instance();

  // Look up an entry by fingerprint. Returns a *copy* of the cached
  // hint on hit (Database mutation across callers is unsafe, copies
  // bound the blast radius). Returns false on miss.
  bool TryGet(uint64_t fp, DoublePirHint* hint_out);

  // Insert / overwrite an entry. Evicts the least-recently-used
  // entry if size() == capacity_.
  void Put(uint64_t fp, DoublePirHint hint);

  // Test / observability helpers.
  std::size_t Size() const;
  std::size_t Capacity() const { return capacity_; }
  uint64_t Hits() const;
  uint64_t Misses() const;
  void Clear();

  // Test-only: bound capacity (lets unit tests force eviction).
  void SetCapacityForTest(std::size_t cap);

  // On-disk persistence — task 5.6 chunk 4. Lets the cache survive a
  // process restart so production callers don't repeat the cold-start
  // hint computation. File format (little-endian throughout):
  //
  //   magic     : "PHHC"   4 bytes  ("PrimiHub Hint Cache")
  //   version   : u16      2 bytes  (current = 1)
  //   reserved  : u16      2 bytes  (must be 0)
  //   count     : u64      8 bytes  (number of entries)
  //   for each entry, in MRU-first order:
  //     fp       : u64                 fingerprint
  //     blob_len : u64                 SerializeHint output length
  //     blob     : blob_len bytes      SerializeHint output (PHHB)
  //
  // SaveToFile writes atomically via `<path>.tmp` + rename, so a
  // partial write never corrupts the destination. Returns FAIL on
  // open/write/rename failure; *err is populated.
  retcode SaveToFile(const std::string& path,
                     std::string* err = nullptr) const;

  // Replaces the current cache contents with the entries in `path`.
  // Validates magic / version / framing; refuses to mutate the cache
  // on any framing error (cache state preserved on FAIL). Entries are
  // re-inserted via Put() so LRU ordering matches the file's
  // MRU-first sequence.
  retcode LoadFromFile(const std::string& path,
                       std::string* err = nullptr);

 private:
  HintCache() = default;
  HintCache(const HintCache&) = delete;
  HintCache& operator=(const HintCache&) = delete;

  mutable std::mutex mu_;
  std::size_t capacity_ = kDefaultCapacity;
  // LRU implemented as list (front = MRU) + map of fp -> list iterator.
  std::list<std::pair<uint64_t, DoublePirHint>> lru_;
  std::unordered_map<uint64_t,
                     typename std::list<std::pair<uint64_t, DoublePirHint>>::iterator>
      index_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

// Convenience wrapper: try cache, fall back to HintGen::Compute, then
// populate cache. Caller still owns the lifetime of `*db` — on hit,
// the caller MUST run db.mutable_data().ScalarAdd(p/2) + db.Squish(10,3)
// itself before issuing queries (the cache does not preserve squished
// DB state). On miss, HintGen::Compute mutates `*db` itself.
//
// Returns the same retcode as HintGen::Compute; `hit_out` (optional)
// is set to true when the answer came from the cache.
retcode GetOrComputeHint(core::Database* db,
                        const core::LweParams& params,
                        DoublePirHint* hint_out,
                        std::string* err,
                        HintGenStats* stats_out = nullptr,
                        bool* hit_out = nullptr);

}  // namespace primihub::pir::double_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_DOUBLE_PIR_HINT_CACHE_H_
