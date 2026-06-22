/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

namespace primihub::pir::ypir {

std::pair<std::uint64_t, std::uint64_t> GetBarrettCrs(std::uint64_t modulus) {
  // q = floor(2^128 / modulus), returned as {cr0 = low 64, cr1 = high 64}.
  // 2^128 = 2^64 * 2^64; write 2^64 = q1*m + r1, then
  // 2^128 = (2^64*q1)*m + 2^64*r1, and 2^64*r1 = q2*m + r2, so
  // floor(2^128/m) = 2^64*q1 + q2 (q2 < 2^64). Hence cr1=q1, cr0=q2.
  const __uint128_t two64 = static_cast<__uint128_t>(1) << 64;
  const std::uint64_t q1 = static_cast<std::uint64_t>(two64 / modulus);
  const std::uint64_t r1 = static_cast<std::uint64_t>(two64 % modulus);
  const std::uint64_t q2 =
      static_cast<std::uint64_t>((static_cast<__uint128_t>(r1) << 64) / modulus);
  return {q2, q1};  // {cr0, cr1}
}

std::uint64_t BarrettRawU64(std::uint64_t input, std::uint64_t cr1,
                            std::uint64_t modulus) {
  const std::uint64_t tmp = static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(input) * cr1) >> 64);
  std::uint64_t res = input - tmp * modulus;  // wrapping
  if (res >= modulus) res -= modulus;
  return res;
}

namespace {

// Mirrors spiral-rs barrett_raw_u128 (multiply-accumulate of the two
// 64-bit limbs of `val` against {cr0, cr1}, then zx - tmp1*modulus).
std::uint64_t BarrettRawU128(__uint128_t val, std::uint64_t cr0,
                             std::uint64_t cr1, std::uint64_t modulus) {
  const std::uint64_t zx = static_cast<std::uint64_t>(val);
  const std::uint64_t zy = static_cast<std::uint64_t>(val >> 64);

  const std::uint64_t prody =
      static_cast<std::uint64_t>((static_cast<__uint128_t>(zx) * cr0) >> 64);
  std::uint64_t carry = prody;

  __uint128_t t = static_cast<__uint128_t>(zx) * cr1;
  std::uint64_t tmp2x = static_cast<std::uint64_t>(t);
  std::uint64_t tmp2y = static_cast<std::uint64_t>(t >> 64);

  __uint128_t s = static_cast<__uint128_t>(tmp2x) + carry;
  std::uint64_t tmp1 = static_cast<std::uint64_t>(s);
  std::uint64_t tmp3 = tmp2y + static_cast<std::uint64_t>(s >> 64);

  t = static_cast<__uint128_t>(zy) * cr0;
  tmp2x = static_cast<std::uint64_t>(t);
  tmp2y = static_cast<std::uint64_t>(t >> 64);
  s = static_cast<__uint128_t>(tmp1) + tmp2x;
  tmp1 = static_cast<std::uint64_t>(s);
  carry = tmp2y + static_cast<std::uint64_t>(s >> 64);

  tmp1 = zy * cr1 + tmp3 + carry;       // wrapping
  return zx - tmp1 * modulus;           // wrapping
}

}  // namespace

std::uint64_t BarrettReductionU128Raw(std::uint64_t modulus, std::uint64_t cr0,
                                      std::uint64_t cr1, __uint128_t val) {
  std::uint64_t reduced = BarrettRawU128(val, cr0, cr1, modulus);
  if (reduced >= modulus) reduced -= modulus;
  return reduced;
}

std::uint64_t MultiplyModular(std::uint64_t a, std::uint64_t b,
                              std::uint64_t modulus, std::uint64_t cr0,
                              std::uint64_t cr1, std::size_t crt_count) {
  if (crt_count == 1) {
    return static_cast<std::uint64_t>(
        (static_cast<__uint128_t>(a) * b) % modulus);
  }
  // crt_count == 2: small CRT modulus, a*b fits u64.
  return BarrettRawU64(a * b, cr1, modulus);
}

}  // namespace primihub::pir::ypir
