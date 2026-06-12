/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_prng — seeded pseudorandom number generator interface for
 * the FrodoPIR C++ port. Wraps a deterministic stream cipher keyed
 * by a 32-byte seed. Used by GenerateLweMatrixFromSeed (chunk 2b-i,
 * see frodo_matrices.h) to expand a public seed into the LWE
 * matrix A; the same seed on both client and server must produce
 * the same A.
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 2b-i — partial port-order #4 (utils::matrices PRNG
 *   helpers). Splits off the seeded-RNG interface from chunk 2b-ii
 *   (OpenSSL ChaCha20 swap-in for security) and chunk 2b-iii
 *   (native ChaCha12 port for upstream byte-for-byte).
 *
 * ====================================================================
 *                           !! WARNING !!
 * ====================================================================
 * This chunk ships a PLACEHOLDER backed by std::mt19937_64. mt19937
 * is NOT a cryptographically secure PRNG; its internal state is
 * recoverable from 624 consecutive 32-bit outputs. This placeholder
 * exists solely to unblock chunk 4 (BaseParams) and the dependent
 * portions of api.rs while we land the cryptographic upgrade.
 *
 * The FrodoPIR security analysis requires the public LWE matrix A
 * to be generated from a CSPRNG so the LWE assumption applies.
 * Using mt19937 here is a security regression vs. upstream Rust
 * (which uses rand_chacha::ChaCha12Rng) and MUST be replaced
 * before any production deployment.
 *
 * Chunk 2b-ii will swap the underlying engine to OpenSSL's
 * ChaCha20 (already a primihub dep via gRPC TLS) keeping the
 * SeededRng API surface unchanged. Callers do not need to
 * change after the swap — every chunk-2b-i call site is
 * future-proof via the abstraction.
 *
 * Chunk 2b-iii may add a pure-C++ ChaCha12 implementation to
 * enable upstream Rust <-> C++ byte-for-byte interop if a real
 * deployment ever needs to mix client and server impls. ChaCha20
 * is sufficient for own-protocol security; only mixed-impl interop
 * pulls us back to ChaCha12.
 *
 * Field-by-field correspondence with upstream Rust:
 *   fn get_seeded_rng(s: [u8; 32]) -> StdRng        ←→
 *       SeededRng(span<uint8_t, 32>) ctor
 *   StdRng::next_u32()                              ←→
 *       SeededRng::NextU32()
 *   StdRng::next_u64()                              ←→
 *       SeededRng::NextU64()
 *
 * Two callers using OUR library WILL produce the same stream from
 * the same seed because mt19937_64 + std::seed_seq is deterministic
 * within a fixed libstdc++ version, but the stream WILL NOT match
 * upstream rand_chacha::ChaCha12Rng's stream from the same seed.
 * Cross-impl interop is impossible in chunk 2b-i; chunk 2b-iii is
 * the only path to fix that.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PRNG_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PRNG_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>

namespace primihub::pir::frodo {

// Seed type matching upstream `rand_core::SeedableRng::Seed = [u8; 32]`
// for rand_chacha::ChaCha12Rng. We keep std::array<uint8_t, 32> so the
// API stays unchanged when chunk 2b-ii swaps the backing engine.
using SeedBytes = std::array<std::uint8_t, 32>;

class SeededRng {
 public:
  // Construct from a 32-byte seed. Two SeededRng instances built
  // from equal seeds produce equal infinite streams. WARNING — see
  // file header — the underlying engine is mt19937_64 and is NOT
  // cryptographically secure; this is a chunk-2b-i placeholder.
  explicit SeededRng(const SeedBytes& seed);

  // Returns the next 32-bit output of the stream. Mirrors upstream
  // `StdRng::next_u32()`. Consecutive calls advance the state by 32
  // bits each.
  std::uint32_t NextU32();

  // Returns the next 64-bit output of the stream. Mirrors upstream
  // `StdRng::next_u64()`. Equivalent to two NextU32 calls combined
  // little-endian (lo = first call, hi << 32 = second), so callers
  // alternating NextU32/NextU64 see a coherent linear stream.
  std::uint64_t NextU64();

 private:
  // The engine is exposed in the header (no PIMPL) because the
  // project's bazel C++11 toolchain doesn't have std::make_unique
  // and PIMPL with forward-declared state needs an out-of-line
  // destructor anyway. Cost: any TU that includes frodo_prng.h
  // pulls in <random>. Chunk 2b-ii will swap the engine field
  // (e.g. to a small OpenSSL ChaCha20 wrapper struct) without
  // changing the public API surface.
  std::mt19937_64 engine_;
  std::uint64_t buffer_ = 0;
  bool buffer_has_hi_ = false;
};

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_PRNG_H_
