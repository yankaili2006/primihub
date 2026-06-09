/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/simple_pir/hint_cache.h"

#include <cstdint>

#include <glog/logging.h>

namespace primihub::pir::simple_pir {

namespace {

constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

inline void Mix(uint64_t* h, uint64_t v) {
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
  const core::Matrix& data = db.data();
  for (uint64_t i = 0; i < data.size(); ++i) {
    Mix(&h, static_cast<uint64_t>(data.data()[i]));
  }
  return h;
}

SimpleHintCache& SimpleHintCache::Instance() {
  static SimpleHintCache cache;
  return cache;
}

bool SimpleHintCache::TryGet(uint64_t fp, SimplePirHint* hint_out) {
  if (hint_out == nullptr) return false;
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(fp);
  if (it == index_.end()) {
    ++misses_;
    return false;
  }
  lru_.splice(lru_.begin(), lru_, it->second);
  *hint_out = it->second->second;
  ++hits_;
  return true;
}

void SimpleHintCache::Put(uint64_t fp, SimplePirHint hint) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = index_.find(fp);
  if (it != index_.end()) {
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

std::size_t SimpleHintCache::Size() const {
  std::lock_guard<std::mutex> lock(mu_);
  return lru_.size();
}

uint64_t SimpleHintCache::Hits() const {
  std::lock_guard<std::mutex> lock(mu_);
  return hits_;
}

uint64_t SimpleHintCache::Misses() const {
  std::lock_guard<std::mutex> lock(mu_);
  return misses_;
}

void SimpleHintCache::Clear() {
  std::lock_guard<std::mutex> lock(mu_);
  lru_.clear();
  index_.clear();
  hits_ = 0;
  misses_ = 0;
}

void SimpleHintCache::SetCapacityForTest(std::size_t cap) {
  std::lock_guard<std::mutex> lock(mu_);
  capacity_ = cap;
  while (lru_.size() > capacity_) {
    index_.erase(lru_.back().first);
    lru_.pop_back();
  }
}

retcode GetOrComputeHint(core::Database* db,
                         const core::LweParams& params,
                         SimplePirHint* hint_out,
                         std::string* err,
                         SimpleHintGenStats* stats_out,
                         bool* hit_out) {
  if (db == nullptr || hint_out == nullptr) {
    if (err) *err = "GetOrComputeHint: db / hint_out must be non-null";
    return retcode::FAIL;
  }
  const uint64_t fp = FingerprintDb(*db, params);
  auto& cache = SimpleHintCache::Instance();
  if (cache.TryGet(fp, hint_out)) {
    if (stats_out != nullptr) {
      stats_out->init_ms = 0.0;
      stats_out->setup_ms = 0.0;
      stats_out->squish_ms = 0.0;
    }
    if (hit_out != nullptr) *hit_out = true;
    return retcode::SUCCESS;
  }
  auto rc = SimpleHintGen::Compute(db, params, hint_out, err, stats_out);
  if (rc != retcode::SUCCESS) return rc;
  cache.Put(fp, *hint_out);
  if (hit_out != nullptr) *hit_out = false;
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
