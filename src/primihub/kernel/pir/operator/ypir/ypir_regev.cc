/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"

namespace primihub::pir::ypir {

PolyMatrixRaw RandomRngRaw(const Params& p, std::size_t rows, std::size_t cols,
                           ChaChaRng& rng) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = rows;
  out.cols = cols;
  out.data.assign(rows * cols * pl, 0);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      std::uint64_t* poly = out.Poly(r, c, pl);
      for (std::size_t z = 0; z < pl; ++z) poly[z] = rng.NextU64() % p.modulus;
    }
  }
  return out;
}

PolyMatrixRaw NoiseRaw(const Params& p, std::size_t rows, std::size_t cols,
                       const DiscreteGaussian& dg, ChaChaRng& rng) {
  const std::size_t pl = p.poly_len;
  PolyMatrixRaw out;
  out.rows = rows;
  out.cols = cols;
  out.data.assign(rows * cols * pl, 0);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) {
      std::uint64_t* poly = out.Poly(r, c, pl);
      for (std::size_t z = 0; z < pl; ++z) {
        const std::uint64_t sampled_val = rng.NextU64();
        poly[z] = dg.Sample(p.modulus, sampled_val);
      }
    }
  }
  return out;
}

}  // namespace primihub::pir::ypir
