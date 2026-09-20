/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_server sub-chunk 10a (GenerateYConstants / SplitAlloc /
 * DbRowsPadded / DbCols).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"

namespace primihub::pir::ypir {
namespace {

// spiral_rs DEFAULT_MODULI (both == 1 mod 2*poly_len for HEXL NTT).
const std::vector<std::uint64_t> kDefaultModuli = {268369921ULL, 249561089ULL};

Params MakeExpansionParams() {
  // nu_1=1, nu_2=1, p=256, q2_bits=28, t_exp_left=2. poly_len fixed 2048.
  return ParamsForExpansion(1, 1, 256, 28, 2, kDefaultModuli);
}

// GenerateYConstants: each y[k] must be NTT( X^(poly_len/2^(k+1)) ), so
// FromNtt(y[k]) is a clean monomial; neg_y[k] is modulus-1 at that slot.
TEST(YpirServerTest, GenerateYConstants_MonomialRoundTrip) {
  Params params = MakeExpansionParams();
  NttContext ctx(params);

  YConstants yc = GenerateYConstants(ctx);
  ASSERT_EQ(yc.y.size(), params.poly_len_log2);
  ASSERT_EQ(yc.neg_y.size(), params.poly_len_log2);

  for (std::size_t k = 0; k < params.poly_len_log2; ++k) {
    const std::size_t num_cts = static_cast<std::size_t>(1) << (k + 1);
    const std::size_t exp = params.poly_len / num_cts;

    PolyMatrixRaw y = ctx.FromNtt(yc.y[k]);
    PolyMatrixRaw ny = ctx.FromNtt(yc.neg_y[k]);
    const std::uint64_t* yp = y.Poly(0, 0, params.poly_len);
    const std::uint64_t* nyp = ny.Poly(0, 0, params.poly_len);

    for (std::size_t i = 0; i < params.poly_len; ++i) {
      if (i == exp) {
        EXPECT_EQ(yp[i], 1ULL) << "k=" << k << " exp=" << exp;
        EXPECT_EQ(nyp[i], params.modulus - 1) << "k=" << k << " exp=" << exp;
      } else {
        EXPECT_EQ(yp[i], 0ULL) << "k=" << k << " i=" << i;
        EXPECT_EQ(nyp[i], 0ULL) << "k=" << k << " i=" << i;
      }
    }
  }
}

// SplitAlloc: 2 rows -> 4 out_rows, 8-bit inputs re-chunked to 4-bit
// (LSB-first). Hand-computed for two columns.
TEST(YpirServerTest, SplitAlloc_RechunksBitstream) {
  // row-major rows=2 cols=2: [r0c0,r0c1, r1c0,r1c1]
  std::vector<std::uint64_t> buf = {0xAB, 0x12, 0xCD, 0x34};
  std::vector<std::uint16_t> out = SplitAlloc(buf,
                                              /*special_bit_offs=*/8,
                                              /*rows=*/2, /*cols=*/2,
                                              /*out_rows=*/4,
                                              /*inp_mod_bits=*/8,
                                              /*pt_bits=*/4);
  std::vector<std::uint16_t> expected = {0xB, 0x2, 0xA, 0x1,
                                         0xD, 0x4, 0xC, 0x3};
  EXPECT_EQ(out, expected);
}

// Single column, natural offset, verifies nibble ordering explicitly.
TEST(YpirServerTest, SplitAlloc_SingleColumnNibbleOrder) {
  std::vector<std::uint64_t> buf = {0xAB, 0xCD};
  std::vector<std::uint16_t> out = SplitAlloc(buf, 8, 2, 1, 4, 8, 4);
  std::vector<std::uint16_t> expected = {0xB, 0xA, 0xD, 0xC};
  EXPECT_EQ(out, expected);
}

TEST(YpirServerTest, DbDimensions) {
  Params p;
  p.poly_len = 2048;
  p.poly_len_log2 = 11;
  p.db_dim_1 = 2;
  p.db_dim_2 = 3;
  p.instances = 4;

  EXPECT_EQ(DbRowsPadded(p, true), static_cast<std::size_t>(1) << (2 + 11));
  EXPECT_EQ(DbRowsPadded(p, false), static_cast<std::size_t>(1) << (2 + 11));
  EXPECT_EQ(DbCols(p, /*is_simplepir=*/true), 4u * 2048u);
  EXPECT_EQ(DbCols(p, /*is_simplepir=*/false),
            static_cast<std::size_t>(1) << (3 + 11));
}

}  // namespace
}  // namespace primihub::pir::ypir
