/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_ops.h"

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

namespace primihub::pir::ypir {

namespace {

// add_modular: (a + b) mod q_m.
inline std::uint64_t AddModular(std::uint64_t a, std::uint64_t b,
                                std::uint64_t modulus, std::uint64_t cr1) {
  return BarrettRawU64(a + b, cr1, modulus);
}

// multiply_add_modular (mirrors spiral_rs incl. its crt_count==1 quirk
// of not adding x on that path; YPIR uses crt_count==2).
inline std::uint64_t MultiplyAddModular(std::uint64_t a, std::uint64_t b,
                                        std::uint64_t x, std::uint64_t modulus,
                                        std::uint64_t cr1,
                                        std::size_t crt_count) {
  if (crt_count == 1) {
    return static_cast<std::uint64_t>(
        (static_cast<__uint128_t>(a) * b) % modulus);
  }
  return BarrettRawU64(a * b + x, cr1, modulus);
}

}  // namespace

PolyMatrixNTT AddNtt(const Params& p, const PolyMatrixNTT& a,
                     const PolyMatrixNTT& b) {
  const std::size_t pl = p.poly_len;
  const std::size_t cc = p.crt_count;
  const std::size_t blk = cc * pl;
  PolyMatrixNTT res;
  res.rows = a.rows;
  res.cols = a.cols;
  res.data.assign(a.rows * a.cols * blk, 0);
  for (std::size_t i = 0; i < a.rows; ++i) {
    for (std::size_t j = 0; j < a.cols; ++j) {
      std::uint64_t* rp = res.Poly(i, j, blk);
      const std::uint64_t* ap = a.Poly(i, j, blk);
      const std::uint64_t* bp = b.Poly(i, j, blk);
      for (std::size_t m = 0; m < cc; ++m) {
        for (std::size_t z = 0; z < pl; ++z) {
          const std::size_t idx = m * pl + z;
          rp[idx] = AddModular(ap[idx], bp[idx], p.moduli[m], p.barrett_cr_1[m]);
        }
      }
    }
  }
  return res;
}

PolyMatrixNTT MultiplyNtt(const Params& p, const PolyMatrixNTT& a,
                          const PolyMatrixNTT& b) {
  const std::size_t pl = p.poly_len;
  const std::size_t cc = p.crt_count;
  const std::size_t blk = cc * pl;
  PolyMatrixNTT res;
  res.rows = a.rows;
  res.cols = b.cols;
  res.data.assign(a.rows * b.cols * blk, 0);
  for (std::size_t i = 0; i < a.rows; ++i) {
    for (std::size_t j = 0; j < b.cols; ++j) {
      std::uint64_t* rp = res.Poly(i, j, blk);  // zero-initialised
      for (std::size_t k = 0; k < a.cols; ++k) {
        const std::uint64_t* ap = a.Poly(i, k, blk);
        const std::uint64_t* bp = b.Poly(k, j, blk);
        for (std::size_t m = 0; m < cc; ++m) {
          for (std::size_t z = 0; z < pl; ++z) {
            const std::size_t idx = m * pl + z;
            rp[idx] = MultiplyAddModular(ap[idx], bp[idx], rp[idx], p.moduli[m],
                                         p.barrett_cr_1[m], cc);
          }
        }
      }
    }
  }
  return res;
}

}  // namespace primihub::pir::ypir
