/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace primihub::pir::ypir {

namespace {
// spiral-rs params.rs Q2_VALUES (menonsamir/spiral-rs@6929441): a 37-entry
// table indexed by q2_bits. Indices 0..13 are 0 (unused); 14 onward are the
// largest NTT-friendly prime < 2^q2_bits congruent to 1 mod 2*poly_len.
constexpr std::uint64_t kQ2Values[37] = {
    0,           0,          0,          0,          0,          0,
    0,           0,          0,          0,          0,          0,
    0,           0,          12289,      12289,      61441,      65537,
    65537,       520193,     786433,     786433,     3604481,    7340033,
    16515073,    33292289,   67043329,   132120577,  268369921,  469762049,
    1073479681,  2013265921, 4293918721, 8588886017, 17175674881,
    34359214081, 68718428161};
}  // namespace

std::array<std::uint8_t, 32> StaticPublicSeed() {
  return std::array<std::uint8_t, 32>{};  // all zeros
}

std::array<std::uint8_t, 32> StaticSeed2() {
  std::array<std::uint8_t, 32> s{};
  s[0] = 2;
  return s;
}

std::array<std::uint8_t, 32> GetSeed(std::uint8_t public_seed_idx) {
  std::array<std::uint8_t, 32> seed = StaticPublicSeed();
  seed[0] = public_seed_idx;
  return seed;
}

std::uint64_t GetQPrime1(const Params& /*params*/) {
  return static_cast<std::uint64_t>(1) << 20;
}

std::uint64_t GetQPrime2(const Params& params) {
  if (params.q2_bits == params.modulus_log2) return params.modulus;
  // Soft boundary: upstream indexes directly (panics on OOB); the valid
  // q2_bits range is [14, 36].
  if (params.q2_bits >= 37) return 0;
  return kQ2Values[params.q2_bits];
}

}  // namespace primihub::pir::ypir
