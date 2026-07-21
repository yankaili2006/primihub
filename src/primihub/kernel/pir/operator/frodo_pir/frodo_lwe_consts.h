/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_lwe_consts — C++ port of upstream brave-experiments/
 * frodo-pir@15573960 src/utils.rs `pub mod lwe`. Three trivial
 * integer-math constants for the LWE rounding boundary used by
 * FrodoPIR queries.
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 1 — first chunk of the FrodoPIR algorithmic port
 *   (task 7.1). Zero dependencies; no Spiral, no upstream Rust
 *   link, no external crate.
 *
 * Field-by-field correspondence with upstream Rust:
 *   const MODULUS: u64 = u32::MAX as u64 + 1     →
 *       kLweModulus = static_cast<uint64_t>(1) << 32
 *   pub fn get_rounding_factor(plaintext_bits: usize) -> u32  →
 *       GetRoundingFactor(plaintext_bits) -> uint32_t
 *   pub fn get_rounding_floor(plaintext_bits: usize) -> u32   →
 *       GetRoundingFloor(plaintext_bits) -> uint32_t
 *   pub fn get_plaintext_size(plaintext_bits: usize) -> u32   →
 *       GetPlaintextSize(plaintext_bits) -> uint32_t
 *
 * All defined as constexpr in the header so callers can use them
 * at compile time (e.g., as template arguments or array sizes).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_LWE_CONSTS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_LWE_CONSTS_H_

#include <cstddef>
#include <cstdint>

namespace primihub::pir::frodo {

// Mirrors upstream: const MODULUS: u64 = u32::MAX as u64 + 1.
// That's 2^32 = 4294967296. The LWE ciphertext space is Z_(2^32).
constexpr std::uint64_t kLweModulus =
    static_cast<std::uint64_t>(1) << 32;

// Plaintext modulus = 2^plaintext_bits. Upstream returns u32, so
// plaintext_bits is implicitly capped at 32; values >= 32 saturate
// to 0 in upstream because `2u32.pow(32)` overflows in debug and
// wraps in release. We return 0 for plaintext_bits >= 32 to be
// explicit about the overflow behavior.
constexpr std::uint32_t GetPlaintextSize(std::size_t plaintext_bits) {
  return (plaintext_bits >= 32)
             ? 0u
             : (static_cast<std::uint32_t>(1) << plaintext_bits);
}

// Rounding factor = MODULUS / plaintext_size. With plaintext_size = 0
// (degenerate) returns 0; upstream would panic on div-by-zero.
constexpr std::uint32_t GetRoundingFactor(std::size_t plaintext_bits) {
  return (GetPlaintextSize(plaintext_bits) == 0u)
             ? 0u
             : static_cast<std::uint32_t>(
                   kLweModulus /
                   static_cast<std::uint64_t>(
                       GetPlaintextSize(plaintext_bits)));
}

// Rounding floor = rounding_factor / 2. Decides whether a bit in
// the queried DB row is 0 (below) or 1 (above).
constexpr std::uint32_t GetRoundingFloor(std::size_t plaintext_bits) {
  return GetRoundingFactor(plaintext_bits) / 2u;
}

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_LWE_CONSTS_H_
