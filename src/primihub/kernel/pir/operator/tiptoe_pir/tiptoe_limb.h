/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Pure limb-decomposition helpers for the Tiptoe hint (ported from the
 * constants + getChunk of underhood/underhood/hint.go), tiptoe chunk 1.1e.
 * Zero-dependency, .50-testable. The hint H (a matrix of 32/64-bit SimplePIR
 * elements) is split into BitsPerLimb-wide limbs so H*s can be computed
 * homomorphically in BFV without overflowing the plaintext modulus.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LIMB_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LIMB_H_

#include <cstdint>

namespace primihub::pir::tiptoe {

// Each hint element is decomposed into 4-bit limbs.
inline constexpr int kBitsPerLimb = 4;

// Only the top limbs are needed to recover the bits SimplePIR decryption uses.
// 64-bit elements (SimplePIR p=2^17): top 8 of 16 limbs. 32-bit (p=2^8): top 5
// of 8. Each limb's homomorphic partial sum maxes at ~2^16 (secret dim 2^11 *
// max secret 2 * max limb 2^4-1), which fits the BFV plaintext modulus 65537.
inline constexpr int kNumLimbs64 = 8;
inline constexpr int kNumLimbs32 = 5;

// Total limbs in an element of the given bit width (16 for 64-bit, 8 for 32).
inline constexpr int MaxLimbs(int elem_bits) { return elem_bits / kBitsPerLimb; }

// Number of (top) limbs actually computed for the given element bit width.
inline constexpr int LimbsFor(int elem_bits) {
  return elem_bits == 64 ? kNumLimbs64 : (elem_bits == 32 ? kNumLimbs32 : 0);
}

// Extract the chunk-th kBitsPerLimb-bit limb of v (underhood getChunk).
inline std::uint64_t GetChunk(std::uint64_t v, int chunk) {
  const std::uint64_t mask = (static_cast<std::uint64_t>(1) << kBitsPerLimb) - 1;
  v &= (mask << (chunk * kBitsPerLimb));
  v >>= (chunk * kBitsPerLimb);
  return v;  // < 2^kBitsPerLimb after masking (Go panics if not; can't here)
}

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_LIMB_H_
