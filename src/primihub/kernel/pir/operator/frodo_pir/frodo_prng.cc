/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SeededRng backed by OpenSSL ChaCha20 (chunk 2b-ii). See
 * frodo_prng.h header for the cryptographic-property checklist
 * and the swap rationale from chunk 2b-i.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace primihub::pir::frodo {

namespace {

// Aborts the process with a diagnostic when an OpenSSL EVP call
// returns a failure. Used in the ctor and refill — both code paths
// where a failure means our crypto invariant is broken and the
// caller's keystream would be unsafe to use, so failing fast is
// safer than continuing with a degraded RNG.
[[noreturn]] void OpensslFatal(const char* where) {
  std::string msg = "frodo_prng: OpenSSL call failed at ";
  msg += where;
  // Throw rather than abort so unit tests can observe and we don't
  // SIGABRT the test binary. SeededRng is constructed only inside
  // GenerateLweMatrixFromSeed which runs early in BaseParams::New;
  // bubbling up gives that path a clean failure.
  throw std::runtime_error(msg);
}

}  // namespace

SeededRng::SeededRng(const SeedBytes& seed) {
  ctx_ = EVP_CIPHER_CTX_new();
  if (ctx_ == nullptr) {
    OpensslFatal("EVP_CIPHER_CTX_new");
  }
  // ChaCha20 expects a 16-byte IV: 4-byte block counter + 12-byte
  // nonce per RFC 8439 (OpenSSL's EVP layer takes them concatenated
  // little-endian, counter first). We use all-zero so the keystream
  // starts at counter=0 — same convention as upstream
  // rand_chacha::ChaCha12Rng::from_seed.
  std::array<std::uint8_t, 16> iv{};
  if (EVP_EncryptInit_ex(ctx_, EVP_chacha20(), nullptr, seed.data(),
                         iv.data()) != 1) {
    EVP_CIPHER_CTX_free(ctx_);
    ctx_ = nullptr;
    OpensslFatal("EVP_EncryptInit_ex(EVP_chacha20)");
  }
  block_pos_ = 64;  // empty — first NextU32/NextU64 will refill
}

SeededRng::~SeededRng() {
  if (ctx_ != nullptr) {
    EVP_CIPHER_CTX_free(ctx_);
    ctx_ = nullptr;
  }
}

void SeededRng::RefillBlock() {
  // Feed 64 zero bytes through ChaCha20 to harvest one 64-byte
  // keystream block. Out-of-place is fine; both buffers must be
  // distinct per EVP contract.
  std::array<std::uint8_t, 64> zeros{};  // value-init -> all zero
  int out_len = 0;
  if (EVP_EncryptUpdate(ctx_, block_.data(), &out_len, zeros.data(),
                        static_cast<int>(zeros.size())) != 1) {
    OpensslFatal("EVP_EncryptUpdate");
  }
  if (out_len != 64) {
    OpensslFatal("EVP_EncryptUpdate-short-write");
  }
  block_pos_ = 0;
}

void SeededRng::ReadBytes(std::uint8_t* out, std::size_t n) {
  // Drain the byte stream in pieces, refilling the buffer as needed.
  // For n in {4, 8} this loops at most twice (once if n bytes fit
  // in the current block, twice if it straddles a block boundary).
  std::size_t taken = 0;
  while (taken < n) {
    if (block_pos_ >= block_.size()) {
      RefillBlock();
    }
    const std::size_t avail = block_.size() - block_pos_;
    const std::size_t want = n - taken;
    const std::size_t chunk = (avail < want) ? avail : want;
    std::memcpy(out + taken, block_.data() + block_pos_, chunk);
    block_pos_ += chunk;
    taken += chunk;
  }
}

std::uint32_t SeededRng::NextU32() {
  std::uint8_t buf[4];
  ReadBytes(buf, 4);
  // Little-endian assembly — matches upstream rand_chacha
  // LeRng::next_u32 which reads the keystream LE.
  return static_cast<std::uint32_t>(buf[0]) |
         (static_cast<std::uint32_t>(buf[1]) << 8) |
         (static_cast<std::uint32_t>(buf[2]) << 16) |
         (static_cast<std::uint32_t>(buf[3]) << 24);
}

std::uint64_t SeededRng::NextU64() {
  std::uint8_t buf[8];
  ReadBytes(buf, 8);
  std::uint64_t v = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    v |= static_cast<std::uint64_t>(buf[i]) << (i * 8);
  }
  return v;
}

void SeededRng::FillBytesBulk(std::uint8_t* out, std::size_t n) {
  if (n == 0) return;
  // Drain whatever bytes remain in the current block first so
  // the stream offset stays consistent with NextU32/NextU64
  // callers that interleave with bulk fills.
  std::size_t taken = 0;
  if (block_pos_ < block_.size()) {
    const std::size_t avail = block_.size() - block_pos_;
    const std::size_t want = (avail < n) ? avail : n;
    std::memcpy(out, block_.data() + block_pos_, want);
    block_pos_ += want;
    taken = want;
  }

  // Bulk-encrypt the entire middle range in ONE EVP call. We
  // zero `out` first so EVP_EncryptUpdate xors zero against
  // the keystream — same effect as the per-64-byte loop, just
  // with two-orders-of-magnitude lower call overhead.
  if (taken < n) {
    const std::size_t bulk = ((n - taken) / 64) * 64;
    if (bulk > 0) {
      std::memset(out + taken, 0, bulk);
      int out_len = 0;
      if (EVP_EncryptUpdate(ctx_, out + taken, &out_len,
                             out + taken,
                             static_cast<int>(bulk)) != 1) {
        OpensslFatal("EVP_EncryptUpdate-bulk");
      }
      if (static_cast<std::size_t>(out_len) != bulk) {
        OpensslFatal("EVP_EncryptUpdate-bulk-short-write");
      }
      taken += bulk;
    }
  }

  // Partial tail (< 64 bytes) — refill a block and memcpy. We
  // cannot use a smaller EVP call here without leaving the
  // block buffer in a wrong-position state for subsequent
  // NextU32 callers.
  if (taken < n) {
    RefillBlock();
    const std::size_t want = n - taken;
    std::memcpy(out + taken, block_.data(), want);
    block_pos_ = want;
  }
}

void SeededRng::FillKeystreamBulk(std::uint8_t* out, std::size_t n) {
  if (n == 0) return;
  // Identical control flow to FillBytesBulk except for the
  // missing memset(out, 0, bulk) in the bulk branch. The
  // contract documented in frodo_prng.h pins this asymmetry:
  // caller swears `out[0..n)` is zero, so skipping the memset
  // simply replaces "write zero then XOR with keystream" with
  // "XOR keystream into already-zero" — same final byte stream
  // when the precondition holds.
  std::size_t taken = 0;
  if (block_pos_ < block_.size()) {
    const std::size_t avail = block_.size() - block_pos_;
    const std::size_t want = (avail < n) ? avail : n;
    std::memcpy(out, block_.data() + block_pos_, want);
    block_pos_ += want;
    taken = want;
  }
  if (taken < n) {
    const std::size_t bulk = ((n - taken) / 64) * 64;
    if (bulk > 0) {
      int out_len = 0;
      if (EVP_EncryptUpdate(ctx_, out + taken, &out_len,
                             out + taken,
                             static_cast<int>(bulk)) != 1) {
        OpensslFatal("EVP_EncryptUpdate-bulk-keystream");
      }
      if (static_cast<std::size_t>(out_len) != bulk) {
        OpensslFatal("EVP_EncryptUpdate-bulk-keystream-short-write");
      }
      taken += bulk;
    }
  }
  if (taken < n) {
    RefillBlock();
    const std::size_t want = n - taken;
    std::memcpy(out + taken, block_.data(), want);
    block_pos_ = want;
  }
}



namespace os_rng {

void FillBytes(std::uint8_t* out, std::size_t n) {
  if (n == 0) {
    return;
  }
  // RAND_bytes returns 1 on success, 0 on failure, -1 if not
  // supported. Treat anything but 1 as fatal — a degraded OsRng
  // would silently weaken every downstream cryptographic primitive.
  const int rc = RAND_bytes(out, static_cast<int>(n));
  if (rc != 1) {
    throw std::runtime_error(
        "frodo_prng::os_rng::FillBytes: RAND_bytes returned non-1 "
        "(possible kernel entropy / RAND_status failure)");
  }
}

std::uint32_t NextU32() {
  std::uint8_t buf[4];
  FillBytes(buf, 4);
  return static_cast<std::uint32_t>(buf[0]) |
         (static_cast<std::uint32_t>(buf[1]) << 8) |
         (static_cast<std::uint32_t>(buf[2]) << 16) |
         (static_cast<std::uint32_t>(buf[3]) << 24);
}

}  // namespace os_rng

SeedBytes GenerateSeed() {
  SeedBytes seed;
  os_rng::FillBytes(seed.data(), seed.size());
  return seed;
}

}  // namespace primihub::pir::frodo
