/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing_fast.h"

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

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

PolyMatrixNTT FastMultiplyNoReduce(const Params& p, const PolyMatrixNTT& a,
                                   const PolyMatrixNTT& b) {
  const std::size_t pl = p.poly_len;
  const std::size_t blk = p.crt_count * pl;
  const std::size_t K = a.cols;
  PolyMatrixNTT res;
  res.rows = 1;
  res.cols = 1;
  res.data.assign(blk, 0);
  std::uint64_t* rp = res.data.data();
  for (std::size_t idx = 0; idx < pl; ++idx) {
    std::uint64_t sum_lo = 0, sum_hi = 0;  // limb 0 / limb 1 accumulators
    for (std::size_t k = 0; k < K; ++k) {
      const std::uint64_t x = a.Poly(0, k, blk)[idx];
      const std::uint64_t y = b.Poly(k, 0, blk)[idx];
      sum_lo += (x & 0xffffffffull) * (y & 0xffffffffull);
      sum_hi += (x >> 32) * (y >> 32);
    }
    rp[idx] = sum_lo;
    rp[pl + idx] = sum_hi;
  }
  return res;
}

void FastReduce(const Params& p, PolyMatrixNTT& res) {
  const std::size_t pl = p.poly_len;
  for (std::size_t m = 0; m < p.crt_count; ++m) {
    for (std::size_t i = 0; i < pl; ++i) {
      const std::size_t idx = m * pl + i;
      res.data[idx] = BarrettRawU64(res.data[idx], p.barrett_cr_1[m], p.moduli[m]);
    }
  }
}

void FastAddInto(const Params& p, PolyMatrixNTT& res, const PolyMatrixNTT& a) {
  const std::size_t pl = p.poly_len;
  const std::size_t blk = p.crt_count * pl;
  for (std::size_t i = 0; i < res.rows; ++i) {
    for (std::size_t j = 0; j < res.cols; ++j) {
      std::uint64_t* rp = res.Poly(i, j, blk);
      const std::uint64_t* ap = a.Poly(i, j, blk);
      for (std::size_t m = 0; m < p.crt_count; ++m) {
        for (std::size_t z = 0; z < pl; ++z) {
          const std::size_t idx = m * pl + z;
          rp[idx] = FastBarrettRawU64(rp[idx] + ap[idx], p.barrett_cr_1[m],
                                      p.moduli[m]);
        }
      }
    }
  }
}

void FastAddIntoNoReduce(PolyMatrixNTT& res, const PolyMatrixNTT& a) {
  for (std::size_t i = 0; i < res.data.size(); ++i) res.data[i] += a.data[i];
}

}  // namespace primihub::pir::ypir
