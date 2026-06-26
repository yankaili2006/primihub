/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace primihub::pir::ypir {

std::uint64_t MultiplyUintMod(std::uint64_t a, std::uint64_t b,
                              std::uint64_t modulus) {
  return static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b)) % modulus);
}

std::uint64_t InvertUintMod(std::uint64_t value, std::uint64_t modulus) {
  if (value == 0) return 0;  // None
  // extended_gcd(value, modulus): track only the Bezout coefficient for value.
  std::uint64_t x = value, y = modulus;
  std::int64_t prev_a = 1, a = 0;
  while (y != 0) {
    const std::int64_t q = static_cast<std::int64_t>(x / y);
    const std::uint64_t t = x % y;
    x = y;
    y = t;
    const std::int64_t ta = a;
    a = prev_a - q * a;
    prev_a = ta;
  }
  if (x != 1) return 0;  // gcd != 1 -> None
  if (prev_a < 0)
    return static_cast<std::uint64_t>(prev_a) + modulus;  // u64 wrap == +modulus
  return static_cast<std::uint64_t>(prev_a);
}

PolyMatrixRaw SingleValue(const Params& params, std::uint64_t value) {
  PolyMatrixRaw r;
  r.rows = 1;
  r.cols = 1;
  r.data.assign(params.poly_len, 0);
  r.data[0] = value;
  return r;
}

PolyMatrixRaw MatrixWithIdentity(const Params& params,
                                 const PolyMatrixRaw& in) {
  assert(in.cols == 1);
  const std::size_t rows = in.rows;
  const std::size_t poly_len = params.poly_len;

  PolyMatrixRaw r;
  r.rows = rows;
  r.cols = rows + 1;
  r.data.assign(rows * (rows + 1) * poly_len, 0);

  // column 0 = in
  for (std::size_t row = 0; row < rows; ++row) {
    const std::uint64_t* src = in.Poly(row, 0, poly_len);
    std::uint64_t* dst = r.Poly(row, 0, poly_len);
    for (std::size_t z = 0; z < poly_len; ++z) dst[z] = src[z];
  }
  // columns 1..=rows = identity (constant 1 on the diagonal)
  for (std::size_t d = 0; d < rows; ++d) r.Poly(d, 1 + d, poly_len)[0] = 1;

  return r;
}

void GenTernaryMat(const Params& params, PolyMatrixRaw& mat,
                   std::size_t hamming, ChaChaRng& rng) {
  assert(2 * hamming <= params.poly_len);
  const std::uint64_t modulus = params.modulus;
  const std::size_t poly_len = params.poly_len;

  for (std::size_t r = 0; r < mat.rows; ++r) {
    for (std::size_t c = 0; c < mat.cols; ++c) {
      std::uint64_t* pol = mat.Poly(r, c, poly_len);
      for (std::size_t z = 0; z < poly_len; ++z) pol[z] = 0;
      for (std::size_t i = 0; i < hamming; ++i) pol[i] = 1;
      for (std::size_t i = hamming; i < 2 * hamming; ++i) pol[i] = modulus - 1;
      // Fisher-Yates shuffle over the poly_len coefficients.
      for (std::size_t i = poly_len - 1; i >= 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.NextU32()) % (i + 1);
        std::swap(pol[i], pol[j]);
      }
    }
  }
}

}  // namespace primihub::pir::ypir
