/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_gadget_test — P4. Automorph hand-check + the exact
 * gadget_invert oracle from spiral-rs gadget.rs.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_gadget.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

Params Pl(std::size_t poly_len) {
  return Params::Init(poly_len, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                      1, 1, 1, 0, 0);
}

TEST(YpirGadgetTest, Automorph_HandComputed) {
  const auto p = Pl(4);
  PolyMatrixRaw a;
  a.rows = 1; a.cols = 1; a.data = {10, 20, 30, 40};  // < modulus
  // t=3: out[(i*3)%4] = (((i*3)/4)%2==0) ? a[i] : modulus-a[i]
  //   i0->rem0 num0 a0; i1->rem3 num0 a1; i2->rem2 num1 -a2; i3->rem1 num2 a3
  const auto r = Automorph(p, a, 3);
  EXPECT_EQ(r.data[0], 10u);
  EXPECT_EQ(r.data[1], 40u);
  EXPECT_EQ(r.data[2], p.modulus - 30u);
  EXPECT_EQ(r.data[3], 20u);
}

TEST(YpirGadgetTest, GadgetInvert_BinaryDecomposition) {
  const auto p = Pl(64);
  const std::size_t log_q = static_cast<std::size_t>(p.modulus_log2);
  PolyMatrixRaw mat;
  mat.rows = 2; mat.cols = 1;
  mat.data.assign(2 * 1 * 64, 0);
  mat.Poly(0, 0, 64)[37] = 3;
  mat.Poly(1, 0, 64)[37] = 6;
  const auto r = GadgetInvert(p, 2 * log_q, mat);
  // binary of 3 = 11 ; of 6 = 110
  EXPECT_EQ(r.Poly(0, 0, 64)[37], 1u);
  EXPECT_EQ(r.Poly(2, 0, 64)[37], 1u);
  EXPECT_EQ(r.Poly(4, 0, 64)[37], 0u);
  EXPECT_EQ(r.Poly(1, 0, 64)[37], 0u);
  EXPECT_EQ(r.Poly(3, 0, 64)[37], 1u);
  EXPECT_EQ(r.Poly(5, 0, 64)[37], 1u);
  EXPECT_EQ(r.Poly(7, 0, 64)[37], 0u);
}

TEST(YpirGadgetTest, BuildGadget_BinaryPowers) {
  const auto p = Pl(64);
  const std::size_t log_q = static_cast<std::size_t>(p.modulus_log2);
  const auto g = BuildGadget(p, 1, log_q);  // 1 x log_q, num_elems=log_q -> bits_per=1
  for (std::size_t j = 0; j < log_q && j < 60; ++j) {
    EXPECT_EQ(g.Poly(0, j, 64)[0], 1ull << j) << "j=" << j;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
