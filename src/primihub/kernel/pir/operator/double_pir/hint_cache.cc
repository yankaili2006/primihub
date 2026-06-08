/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/double_pir/hint_cache.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include <glog/logging.h>

#include "src/primihub/kernel/pir/operator/double_pir/hint_serialize.h"

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

// On-disk cache file framing constants (task 5.6 chunk 4). Mirrors the
// hint blob's PHHB style but with a distinct PHHC magic so a hint blob
// file and a cache file can't be cross-loaded by accident.
constexpr char kCacheMagic[4] = {'P', 'H', 'H', 'C'};
constexpr uint16_t kCacheVersion = 1;
// Hard cap on per-entry blob length. Cells already capped by the
// serializer at 1<<32; the practical worst case is ~5 * 16 GiB. Pick
// 256 GiB as a sanity ceiling — anything past that is almost certainly
// a malformed file.
constexpr uint64_t kMaxEntryBlobBytes = static_cast<uint64_t>(256) << 30;

inline void PutU16LE(uint16_t v, std::string* out) {
  uint8_t buf[2] = {static_cast<uint8_t>(v & 0xff),
                    static_cast<uint8_t>((v >> 8) & 0xff)};
  out->append(reinterpret_cast<const char*>(buf), 2);
}

inline void PutU64LE(uint64_t v, std::string* out) {
  uint8_t buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
  }
  out->append(reinterpret_cast<const char*>(buf), 8);
}

inline bool GetU16LE(const std::string& s, size_t* off, uint16_t* v) {
  if (*off + 2 > s.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data() + *off);
  *v = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  *off += 2;
  return true;
}

inline bool GetU64LE(const std::string& s, size_t* off, uint64_t* v) {
  if (*off + 8 > s.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data() + *off);
  *v = 0;
  for (int i = 0; i < 8; ++i) {
    *v |= static_cast<uint64_t>(p[i]) << (i * 8);
  }
  *off += 8;
  return true;
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

retcode HintCache::SaveToFile(const std::string& path,
                              std::string* err) const {
  // Snapshot under lock so the file reflects a consistent point-in-time
  // view of the cache.
  std::string buf;
  buf.append(kCacheMagic, 4);
  PutU16LE(kCacheVersion, &buf);
  PutU16LE(0, &buf);  // reserved
  {
    std::lock_guard<std::mutex> lock(mu_);
    PutU64LE(static_cast<uint64_t>(lru_.size()), &buf);
    for (const auto& entry : lru_) {
      std::string blob;
      std::string blob_err;
      if (SerializeHint(entry.second, &blob, &blob_err) != retcode::SUCCESS) {
        if (err) {
          std::ostringstream oss;
          oss << "HintCache::SaveToFile: SerializeHint failed for fp=0x"
              << std::hex << entry.first << ": " << blob_err;
          *err = oss.str();
        }
        return retcode::FAIL;
      }
      PutU64LE(entry.first, &buf);
      PutU64LE(static_cast<uint64_t>(blob.size()), &buf);
      buf.append(blob);
    }
  }

  // Atomic write: write to <path>.tmp, then rename. This guarantees the
  // destination is either the old contents or the full new contents —
  // never a torn write.
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::SaveToFile: open " << tmp
            << " failed: " << std::strerror(errno);
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    if (!out.good()) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::SaveToFile: write " << tmp
            << " failed: " << std::strerror(errno);
        *err = oss.str();
      }
      out.close();
      std::remove(tmp.c_str());
      return retcode::FAIL;
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    if (err) {
      std::ostringstream oss;
      oss << "HintCache::SaveToFile: rename " << tmp << " -> " << path
          << " failed: " << std::strerror(errno);
      *err = oss.str();
    }
    std::remove(tmp.c_str());
    return retcode::FAIL;
  }
  return retcode::SUCCESS;
}

retcode HintCache::LoadFromFile(const std::string& path, std::string* err) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    if (err) {
      std::ostringstream oss;
      oss << "HintCache::LoadFromFile: open " << path
          << " failed: " << std::strerror(errno);
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string data = ss.str();

  // Validate framing into a local list FIRST. Only swap into the cache
  // on full success so a malformed file never clobbers existing state.
  size_t off = 0;
  if (data.size() < 4) {
    if (err) *err = "HintCache::LoadFromFile: file shorter than magic";
    return retcode::FAIL;
  }
  if (std::memcmp(data.data(), kCacheMagic, 4) != 0) {
    if (err) *err = "HintCache::LoadFromFile: bad magic (expected PHHC)";
    return retcode::FAIL;
  }
  off += 4;
  uint16_t version = 0;
  uint16_t reserved = 0;
  if (!GetU16LE(data, &off, &version) || !GetU16LE(data, &off, &reserved)) {
    if (err) *err = "HintCache::LoadFromFile: truncated header";
    return retcode::FAIL;
  }
  if (version != kCacheVersion) {
    if (err) {
      std::ostringstream oss;
      oss << "HintCache::LoadFromFile: unsupported version " << version
          << " (this build understands " << kCacheVersion << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (reserved != 0) {
    if (err) *err = "HintCache::LoadFromFile: nonzero reserved field";
    return retcode::FAIL;
  }
  uint64_t count = 0;
  if (!GetU64LE(data, &off, &count)) {
    if (err) *err = "HintCache::LoadFromFile: truncated entry count";
    return retcode::FAIL;
  }
  std::vector<std::pair<uint64_t, DoublePirHint>> staged;
  staged.reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t fp = 0, blob_len = 0;
    if (!GetU64LE(data, &off, &fp) || !GetU64LE(data, &off, &blob_len)) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::LoadFromFile: truncated entry header at index " << i;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    if (blob_len > kMaxEntryBlobBytes) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::LoadFromFile: entry " << i
            << " blob_len " << blob_len << " exceeds sanity cap";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    if (off + blob_len > data.size()) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::LoadFromFile: truncated entry body at index " << i
            << " (need " << blob_len << " bytes, have "
            << (data.size() - off) << ")";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    const std::string blob = data.substr(off, blob_len);
    off += blob_len;
    DoublePirHint hint;
    std::string blob_err;
    if (DeserializeHint(blob, &hint, &blob_err) != retcode::SUCCESS) {
      if (err) {
        std::ostringstream oss;
        oss << "HintCache::LoadFromFile: entry " << i
            << " DeserializeHint failed: " << blob_err;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    staged.emplace_back(fp, std::move(hint));
  }
  if (off != data.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "HintCache::LoadFromFile: " << (data.size() - off)
          << " trailing bytes after last entry";
      *err = oss.str();
    }
    return retcode::FAIL;
  }

  // Framing checks all passed — swap into cache. Clear first so the
  // file fully owns the resulting LRU contents; iterate back-to-front
  // and Put each, so the original MRU-first sequence lands as the
  // current LRU's MRU-first sequence.
  Clear();
  for (auto it = staged.rbegin(); it != staged.rend(); ++it) {
    Put(it->first, std::move(it->second));
  }
  return retcode::SUCCESS;
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
