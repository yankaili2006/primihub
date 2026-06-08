/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_cache.h"

#include <cstdint>

#include <glog/logging.h>

namespace primihub::pir::double_pir {

namespace {

// FNV-1a 64 — fast, allocation-free, low collision risk for the DB
// sizes we'll touch. NOT cryptographic — the cache is process-local
// and the consequence of a collision is only a wrong-hint cache hit,
// which surfaces as a recover-byte mismatch (loud).
constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

inline void Mix(uint64_t* h, uint64_t v) {
  // 8-byte chunk feed into FNV-1a.
  for (int i = 0; i < 8; ++i) {
    *h ^= (v >> (i * 8)) & 0xFF;
    *h *= kFnvPrime;
  }
}

}  // namespace

uint64_t FingerprintDb(const core::Database& db,
                       const core::LweParams& params) {
  uint64_t h = kFnvOffset;
  Mix(&h, params.l);
  Mix(&h, params.m);
  Mix(&h, params.p);
  Mix(&h, params.logq);
  Mix(&h, params.n);
  // Fold the DB bytes — uint32 cells, walk linearly. For very large
  // DBs this is O(L*M) but still much cheaper than Setup (Setup is
  // O(L*M*n)).
  const core::Matrix& data = db.data();
  for (uint64_t i = 0; i < data.size(); ++i) {
    Mix(&h, static_cast<uint64_t>(data.data()[i]));
  }
  return h;
}

HintCache& HintCache::Instance() {
  static HintCache cache;
  return cache;
}

bool HintCache::TryGet(uint64_t fp, DoublePirHint* hint_out) {
  if (hint_out == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(fp);
  if (it == index_.end()) {
    ++misses_;
    return false;
  }
  // Promote to front (MRU).
  lru_.splice(lru_.begin(), lru_, it->second);
  *hint_out = it->second->second;  // copy
  ++hits_;
  return true;
}

void HintCache::Put(uint64_t fp, DoublePirHint hint) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(fp);
  if (it != index_.end()) {
    // Overwrite + promote.
    it->second->second = std::move(hint);
    lru_.splice(lru_.begin(), lru_, it->second);
    return;
  }
  lru_.emplace_front(fp, std::move(hint));
  index_[fp] = lru_.begin();
  while (lru_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

std::size_t HintCache::Size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lru_.size();
}

uint64_t HintCache::Hits() const {
  std::lock_guard<std::mutex> lock(mu_);
  return hits_;
}

uint64_t HintCache::Misses() const {
  std::lock_guard<std::mutex> lock(mu_);
  return misses_;
}

void HintCache::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  lru_.clear();
  index_.clear();
  hits_ = 0;
  misses_ = 0;
}

void HintCache::SetCapacityForTest(std::size_t cap) {
  std::lock_guard<std::mutex> lock(mu_);
  capacity_ = cap;
  while (lru_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

retcode GetOrComputeHint(core::Database* db,
                        const core::LweParams& params,
                        DoublePirHint* hint_out,
                        std::string* err,
                        HintGenStats* stats_out,
                        bool* hit_out) {
  if (db == nullptr || hint_out == nullptr) {
    if (err) *err = "GetOrComputeHint: db / hint_out must be non-null";
    return retcode::FAIL;
  }
  const uint64_t fp = FingerprintDb(*db, params);
  auto& cache = HintCache::Instance();
  if (cache.TryGet(fp, hint_out)) {
    if (stats_out != nullptr) {
      // Cache hit — no init/setup work done. Caller can still see
      // the zero-cost timings to distinguish hit from miss.
      stats_out->init_ms = 0.0;
      stats_out->setup_ms = 0.0;
    }
    if (hit_out != nullptr) *hit_out = true;
    return retcode::SUCCESS;
  }
  // Miss — compute, then store. HintGen mutates *db (Squish), so on
  // a future hit the caller's fresh db will fingerprint identically
  // (caller must pass an un-squished db each time).
  auto rc = HintGen::Compute(db, params, hint_out, err, stats_out);
  if (rc != retcode::SUCCESS) return rc;
  cache.Put(fp, *hint_out);
  if (hit_out != nullptr) *hit_out = false;
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::double_pir
