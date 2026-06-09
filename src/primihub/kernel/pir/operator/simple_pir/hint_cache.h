/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::simple_pir::SimpleHintCache — process-local LRU
 * cache for SimplePirHint, keyed by (l, m, p, logq, FNV-1a(DB)).
 *
 * Same shape and tradeoffs as DoublePIR's HintCache (task 5.6 chunk 2,
 * commit 7b017575). Cache hit skips the O(L·M·n) Setup multiply; caller
 * runs the cheap O(L·M) re-Squish locally to restore db state Answer
 * expects (the cache stores hint matrices only, not the squished DB).
 *
 * Thread safety: instance methods take a std::mutex internally.
 * Singleton via Meyers local static.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_CACHE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_CACHE_H_

#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>

#include "src/primihub/kernel/pir/operator/simple_pir/hint_gen.h"

namespace primihub::pir::simple_pir {

// Cheap FNV-1a 64-bit fingerprint over (l,m,p,logq,n) + db bytes.
// Same allocator-free pattern as DoublePIR's FingerprintDb.
uint64_t FingerprintDb(const core::Database& db, const core::LweParams& params);

class SimpleHintCache {
 public:
  static constexpr std::size_t kDefaultCapacity = 16;

  static SimpleHintCache& Instance();

  bool TryGet(uint64_t fp, SimplePirHint* hint_out);
  void Put(uint64_t fp, SimplePirHint hint);

  std::size_t Size() const;
  std::size_t Capacity() const { return capacity_; }
  uint64_t Hits() const;
  uint64_t Misses() const;
  void Clear();
  void SetCapacityForTest(std::size_t cap);

 private:
  SimpleHintCache() = default;
  SimpleHintCache(const SimpleHintCache&) = delete;
  SimpleHintCache& operator=(const SimpleHintCache&) = delete;

  mutable std::mutex mu_;
  std::size_t capacity_ = kDefaultCapacity;
  std::list<std::pair<uint64_t, SimplePirHint>> lru_;
  std::unordered_map<uint64_t,
                     typename std::list<std::pair<uint64_t, SimplePirHint>>::iterator>
      index_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

// Convenience wrapper. On cache hit, the caller MUST re-run the cheap
// Squish pair locally before issuing queries — the cache doesn't store
// the squished DB. On miss, SimpleHintGen::Compute mutates *db (Setup's
// +p/2 + Squish), so a future hit reusing the same fresh db will
// fingerprint identically.
retcode GetOrComputeHint(core::Database* db,
                         const core::LweParams& params,
                         SimplePirHint* hint_out,
                         std::string* err,
                         SimpleHintGenStats* stats_out = nullptr,
                         bool* hit_out = nullptr);

}  // namespace primihub::pir::simple_pir
#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SIMPLE_PIR_HINT_CACHE_H_
