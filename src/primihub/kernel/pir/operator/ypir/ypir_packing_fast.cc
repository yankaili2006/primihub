/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing_fast.h"

#include <algorithm>
#include <utility>
#include <vector>

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

namespace {

// In-place swap of the two halves of [lo, hi) (mirrors swap_midpoint).
// The range length must be even (upstream relies on equal halves).
void SwapMidpoint(std::vector<std::size_t>& cur, std::size_t lo,
                  std::size_t hi) {
  const std::size_t half = (hi - lo) / 2;
  for (std::size_t i = 0; i < half; ++i) {
    std::swap(cur[lo + i], cur[lo + half + i]);
  }
}

}  // namespace

std::vector<std::size_t> ProduceTable(std::size_t poly_len,
                                      std::size_t chunk_size) {
  std::vector<std::size_t> cur(poly_len);
  for (std::size_t i = 0; i < poly_len; ++i) cur[i] = i;
  const std::size_t outer = poly_len / (chunk_size / 2);
  bool do_it = true;
  for (std::size_t os = 0; os < poly_len; os += outer) {
    if (!do_it) {
      do_it = true;
      continue;
    }
    do_it = false;
    const std::size_t oend = std::min(os + outer, poly_len);
    for (std::size_t cs = os; cs < oend; cs += chunk_size) {
      const std::size_t clen = std::min(chunk_size, oend - cs);
      std::size_t offs = 0;
      std::size_t to_add = std::min(chunk_size / 2, clen / 2);
      while (to_add > 0) {
        SwapMidpoint(cur, cs + offs, cs + clen);
        offs += to_add;
        to_add /= 2;
      }
    }
  }
  return cur;
}

std::vector<std::vector<std::size_t>> AutomorphNttTables(
    std::size_t poly_len, std::size_t log2_poly_len) {
  std::vector<std::vector<std::size_t>> tables;
  tables.reserve(log2_poly_len);
  for (std::size_t i = 0; i < log2_poly_len; ++i) {
    const std::size_t chunk_size = static_cast<std::size_t>(1) << i;
    tables.push_back(ProduceTable(poly_len, 2 * chunk_size));
  }
  return tables;
}

}  // namespace primihub::pir::ypir
