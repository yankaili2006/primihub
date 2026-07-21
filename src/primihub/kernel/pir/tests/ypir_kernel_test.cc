/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_kernel chunk 10e: CrtCompose2 + the scalar
 * FastBatchedDotProduct. Oracles check CRT residues directly (independent of
 * crt_compose's internals): the composed value r satisfies r mod q0 == x and
 * r mod q1 == y; the dot-product result's residues equal the independently
 * summed per-limb dot products. Pure scalar logic, no HEXL.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_kernel.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params P8() {
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, 1, 1, 1, 0,
                      0);
}

TEST(YpirKernelTest, CrtCompose2Residues) {
  const Params p = P8();
  const std::uint64_t q0 = p.moduli[0], q1 = p.moduli[1];
  const std::uint64_t xs[] = {0, 1, 5, 12345, q0 - 1};
  const std::uint64_t ys[] = {0, 1, 7, 99999, q1 - 1};
  for (std::uint64_t x : xs) {
    for (std::uint64_t y : ys) {
      const std::uint64_t r = CrtCompose2(p, x, y);
      EXPECT_EQ(r % q0, x) << "x=" << x << " y=" << y;
      EXPECT_EQ(r % q1, y) << "x=" << x << " y=" << y;
    }
  }
}

TEST(YpirKernelTest, FastBatchedDotProductResidues) {
  const Params p = P8();
  const std::uint64_t q0 = p.moduli[0], q1 = p.moduli[1];
  const std::size_t K = 2, a_elems = 16, b_cols = 3;
  const std::size_t b_rows = a_elems;

  // Condensed query a (K * a_elems): each u64 packs lo (limb0) | hi (limb1).
  std::vector<std::uint64_t> a(K * a_elems);
  for (std::size_t i = 0; i < a.size(); ++i) {
    const std::uint64_t lo = (i * 7919u + 13u) % q0;
    const std::uint64_t hi = (i * 104729u + 7u) % q1;
    a[i] = lo | (hi << 32);
  }
  // Transposed db b (b_rows * b_cols), uint16 elements.
  std::vector<std::uint16_t> b(b_rows * b_cols);
  for (std::size_t i = 0; i < b.size(); ++i)
    b[i] = static_cast<std::uint16_t>((i * 37u + 11u) & 0xFFFFu);

  std::vector<std::uint64_t> c(K * b_cols, 0);
  FastBatchedDotProduct<std::uint16_t>(p, K, c.data(), a.data(), a_elems,
                                       b.data(), b_rows, b_cols);

  for (std::size_t k = 0; k < K; ++k) {
    for (std::size_t j = 0; j < b_cols; ++j) {
      __uint128_t tlo = 0, thi = 0;
      for (std::size_t kk = 0; kk < a_elems; ++kk) {
        const std::uint64_t av = a[k * a_elems + kk];
        tlo += static_cast<__uint128_t>(av & 0xFFFFFFFFull) * b[j * b_rows + kk];
        thi += static_cast<__uint128_t>(av >> 32) * b[j * b_rows + kk];
      }
      const std::uint64_t exp_lo = static_cast<std::uint64_t>(tlo % q0);
      const std::uint64_t exp_hi = static_cast<std::uint64_t>(thi % q1);
      EXPECT_EQ(c[k * b_cols + j] % q0, exp_lo) << "k=" << k << " j=" << j;
      EXPECT_EQ(c[k * b_cols + j] % q1, exp_hi) << "k=" << k << " j=" << j;
    }
  }

  // Accumulation: a second call doubles each residue (mod q).
  std::vector<std::uint64_t> c2 = c;
  FastBatchedDotProduct<std::uint16_t>(p, K, c2.data(), a.data(), a_elems,
                                       b.data(), b_rows, b_cols);
  for (std::size_t i = 0; i < c.size(); ++i) {
    EXPECT_EQ(c2[i] % q0, (2 * (c[i] % q0)) % q0) << "i=" << i;
    EXPECT_EQ(c2[i] % q1, (2 * (c[i] % q1)) % q1) << "i=" << i;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
