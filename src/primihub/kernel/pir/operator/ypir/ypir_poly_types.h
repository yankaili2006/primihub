/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_poly_types — the PolyMatrixRaw / PolyMatrixNTT data containers,
 * split out from the HEXL-linked NttContext (ypir_poly.h) so the pure
 * NTT-domain arithmetic in ypir_poly_ops can use them without pulling
 * in HEXL. Part of P2/P3 of the spiral_rs PolyMatrix port.
 *
 * Layout (mirrors spiral_rs):
 *   raw  data: rows*cols*poly_len u64; poly(r,c) at (r*cols+c)*poly_len.
 *   ntt  data: rows*cols*crt_count*poly_len; poly(r,c) modulus m at
 *              (r*cols+c)*crt_count*poly_len + m*poly_len.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_TYPES_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

struct PolyMatrixRaw {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint64_t> data;  // rows*cols*poly_len
  std::uint64_t* Poly(std::size_t r, std::size_t c, std::size_t poly_len) {
    return data.data() + (r * cols + c) * poly_len;
  }
  const std::uint64_t* Poly(std::size_t r, std::size_t c,
                            std::size_t poly_len) const {
    return data.data() + (r * cols + c) * poly_len;
  }
};

struct PolyMatrixNTT {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<std::uint64_t> data;  // rows*cols*crt_count*poly_len
  // Pointer to poly(r,c)'s crt_count*poly_len block.
  std::uint64_t* Poly(std::size_t r, std::size_t c, std::size_t cc_pl) {
    return data.data() + (r * cols + c) * cc_pl;
  }
  const std::uint64_t* Poly(std::size_t r, std::size_t c,
                            std::size_t cc_pl) const {
    return data.data() + (r * cols + c) * cc_pl;
  }
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_TYPES_H_
