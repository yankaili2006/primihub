/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Pure secret-key helpers ported from underhood/underhood/secret.go (tiptoe
 * chunk 1.1c). These are the zero-dependency arithmetic pieces (constants +
 * fromModuloP + inRange); the rlwe-dependent Client::encryptSecret lands with
 * the client layer (chunk 1.1d). See docs/pir/tiptoe-port-plan.md.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_SECRET_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_SECRET_H_

#include <cstdint>

namespace primihub::pir::tiptoe {

// The inner (SimplePIR) encryption scheme uses ternary secrets in {0, 1, 2}
// (underhood secret.go SecretMin/SecretMax). The secret column vector's entries
// must lie in this range before LHE-encryption.
inline constexpr std::uint64_t kSecretMin = 0;
inline constexpr std::uint64_t kSecretMax = 2;

// Map a value v in [0, p) to its centered signed representative, re-expressed
// in T (mirrors underhood secret.go fromModuloP): values above p/2 wrap to the
// negative side (v - p). Caller guarantees v < p. Used to decode plaintext
// coefficients (which live mod p) back to signed SimplePIR matrix elements.
template <typename T>
inline T FromModuloP(std::uint64_t p, std::uint64_t v) {
  if (v > p / 2) {
    return static_cast<T>(static_cast<std::int64_t>(v) -
                          static_cast<std::int64_t>(p));
  }
  return static_cast<T>(v);
}

// Whether a secret entry is a valid (non-negative ternary) value in
// [kSecretMin, kSecretMax] (underhood secret.go inRange). Works because secret
// values are never negative.
template <typename T>
inline bool InRange(T val) {
  return val >= static_cast<T>(kSecretMin) && val <= static_cast<T>(kSecretMax);
}

}  // namespace primihub::pir::tiptoe

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_TIPTOE_PIR_TIPTOE_SECRET_H_
