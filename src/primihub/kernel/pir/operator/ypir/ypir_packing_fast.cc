/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing_fast.h"

namespace primihub::pir::ypir {

std::uint64_t FastBarrettRawU64(std::uint64_t input, std::uint64_t cr1,
                                std::uint64_t modulus) {
  const std::uint64_t tmp = static_cast<std::uint64_t>(
      (static_cast<__uint128_t>(input) * cr1) >> 64);
  return input - tmp * modulus;  // wrapping; no final correction (lazy)
}

PolyMatrixNTT CondenseMatrix(const Params& p, const PolyMatrixNTT& a) {
  const std::size_t pl = p.poly_len;
  const std::size_t blk = p.crt_count * pl;
  PolyMatrixNTT res;
  res.rows = a.rows;
  res.cols = a.cols;
  res.data.assign(a.rows * a.cols * blk, 0);
  for (std::size_t i = 0; i < a.rows; ++i) {
    for (std::size_t j = 0; j < a.cols; ++j) {
      const std::uint64_t* ap = a.Poly(i, j, blk);
      std::uint64_t* rp = res.Poly(i, j, blk);
      for (std::size_t z = 0; z < pl; ++z) {
        rp[z] = ap[z] | (ap[z + pl] << 32);
      }
    }
  }
  return res;
}

PolyMatrixNTT UncondenseMatrix(const Params& p, const PolyMatrixNTT& a) {
  const std::size_t pl = p.poly_len;
  const std::size_t blk = p.crt_count * pl;
  PolyMatrixNTT res;
  res.rows = a.rows;
  res.cols = a.cols;
  res.data.assign(a.rows * a.cols * blk, 0);
  for (std::size_t i = 0; i < a.rows; ++i) {
    for (std::size_t j = 0; j < a.cols; ++j) {
      const std::uint64_t* ap = a.Poly(i, j, blk);
      std::uint64_t* rp = res.Poly(i, j, blk);
      for (std::size_t z = 0; z < pl; ++z) {
        rp[z] = ap[z] & ((1ull << 32) - 1);
        rp[z + pl] = ap[z] >> 32;
      }
    }
  }
  return res;
}

}  // namespace primihub::pir::ypir
