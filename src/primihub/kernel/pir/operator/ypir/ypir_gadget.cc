/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_gadget.h"

namespace primihub::pir::ypir {

PolyMatrixRaw Automorph(const Params& p, const PolyMatrixRaw& a,
                        std::size_t t) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = a.rows;
  out.cols = a.cols;
  out.data.assign(a.rows * a.cols * pl, 0);
  for (std::size_t r = 0; r < a.rows; ++r) {
    for (std::size_t c = 0; c < a.cols; ++c) {
      const std::uint64_t* src = a.Poly(r, c, pl);
      std::uint64_t* dst = out.Poly(r, c, pl);
      for (std::size_t i = 0; i < pl; ++i) {
        const std::size_t num = (i * t) / pl;
        const std::size_t rem = (i * t) % pl;
        dst[rem] = (num % 2 == 0) ? src[i] : (p.modulus - src[i]);
      }
    }
  }
  return out;
}

std::size_t GetBitsPer(const Params& p, std::size_t dim) {
  const std::uint64_t modulus_log2 = p.modulus_log2;
  if (static_cast<std::uint64_t>(dim) == modulus_log2) return 1;
  return static_cast<std::size_t>(modulus_log2 / dim) + 1;
}

PolyMatrixRaw BuildGadget(const Params& p, std::size_t rows, std::size_t cols) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw g;
  g.rows = rows;
  g.cols = cols;
  g.data.assign(rows * cols * pl, 0);
  const std::size_t nx = rows;
  const std::size_t num_elems = cols / nx;  // cols % nx == 0 by contract
  const std::size_t bits_per = GetBitsPer(p, num_elems);
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < num_elems; ++j) {
      if (bits_per * j >= 64) continue;
      g.Poly(i, i + j * nx, pl)[0] = 1ull << (bits_per * j);
    }
  }
  return g;
}

PolyMatrixRaw GadgetInvertRdim(const Params& p, std::size_t mx,
                               const PolyMatrixRaw& inp, std::size_t rdim) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = mx;
  out.cols = inp.cols;
  out.data.assign(mx * inp.cols * pl, 0);
  const std::size_t num_elems = mx / rdim;
  const std::size_t bits_per = GetBitsPer(p, num_elems);
  const std::uint64_t mask = (1ull << bits_per) - 1;
  for (std::size_t j = 0; j < rdim; ++j) {
    for (std::size_t i = 0; i < inp.cols; ++i) {
      const std::uint64_t* src = inp.Poly(j, i, pl);
      for (std::size_t z = 0; z < pl; ++z) {
        const std::uint64_t val = src[z];
        for (std::size_t k = 0; k < num_elems; ++k) {
          const std::size_t bit_offs = k * bits_per;
          const std::uint64_t piece =
              (bit_offs >= 64) ? 0 : ((val >> bit_offs) & mask);
          out.Poly(j + k * rdim, i, pl)[z] = piece;
        }
      }
    }
  }
  return out;
}

PolyMatrixRaw GadgetInvert(const Params& p, std::size_t mx,
                           const PolyMatrixRaw& inp) {
  return GadgetInvertRdim(p, mx, inp, inp.rows);
}

}  // namespace primihub::pir::ypir
