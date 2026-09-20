/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_params_test — P1. Verifies the derived Params fields by internal
 * consistency + cross-check against the P0 barrett helpers (no spiral_rs
 * golden values needed: the relations pin correctness).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

namespace primihub::pir::ypir {
namespace {

constexpr std::uint64_t kQ0 = 268369921ull;
constexpr std::uint64_t kQ1 = 249561089ull;

Params MakeParams() {
  return Params::Init(/*poly_len=*/2048, {kQ0, kQ1}, /*noise_width=*/6.4,
                      /*n=*/1, /*pt_modulus=*/256, /*q2_bits=*/28,
                      /*t_conv=*/4, /*t_exp_left=*/2, /*t_exp_right=*/2,
                      /*t_gsw=*/3, /*expand_queries=*/true, /*db_dim_1=*/1,
                      /*db_dim_2=*/2, /*instances=*/1, /*db_item_size=*/100,
                      /*version=*/0);
}

TEST(YpirParamsTest, DerivedFields) {
  const auto p = MakeParams();
  EXPECT_EQ(p.poly_len, 2048u);
  EXPECT_EQ(p.poly_len_log2, 11u);
  EXPECT_EQ(p.crt_count, 2u);
  EXPECT_EQ(p.moduli[0], kQ0);
  EXPECT_EQ(p.moduli[1], kQ1);
  EXPECT_EQ(p.modulus, kQ0 * kQ1);
  EXPECT_EQ(p.modulus_log2, 56u);  // ceil(log2(268369921*249561089))
}

TEST(YpirParamsTest, BarrettConstants_MatchP0) {
  const auto p = MakeParams();
  for (int i = 0; i < 2; ++i) {
    const auto crs = GetBarrettCrs(p.moduli[i]);
    EXPECT_EQ(p.barrett_cr_0[i], crs.first) << "i=" << i;
    EXPECT_EQ(p.barrett_cr_1[i], crs.second) << "i=" << i;
  }
  const auto crs_mod = GetBarrettCrs(p.modulus);
  EXPECT_EQ(p.barrett_cr_0_modulus, crs_mod.first);
  EXPECT_EQ(p.barrett_cr_1_modulus, crs_mod.second);
}

TEST(YpirParamsTest, CrtIdempotents) {
  const auto p = MakeParams();
  // mod0_inv_mod1 == 1 (mod q1), 0 (mod q0); mod1_inv_mod0 the mirror.
  EXPECT_EQ(p.mod0_inv_mod1 % kQ1, 1u);
  EXPECT_EQ(p.mod0_inv_mod1 % kQ0, 0u);
  EXPECT_EQ(p.mod1_inv_mod0 % kQ0, 1u);
  EXPECT_EQ(p.mod1_inv_mod0 % kQ1, 0u);
}

TEST(YpirParamsTest, PassthroughFields) {
  const auto p = MakeParams();
  EXPECT_EQ(p.n, 1u);
  EXPECT_EQ(p.pt_modulus, 256u);
  EXPECT_EQ(p.q2_bits, 28u);
  EXPECT_EQ(p.t_conv, 4u);
  EXPECT_EQ(p.t_gsw, 3u);
  EXPECT_TRUE(p.expand_queries);
  EXPECT_EQ(p.db_dim_2, 2u);
  EXPECT_EQ(p.db_item_size, 100u);
}

TEST(YpirParamsTest, SingleModulus) {
  const auto p = Params::Init(2048, {kQ0}, 6.4, 1, 256, 28, 4, 2, 2, 3, true,
                              1, 1, 1, 0, 0);
  EXPECT_EQ(p.crt_count, 1u);
  EXPECT_EQ(p.modulus, kQ0);
  EXPECT_EQ(p.mod0_inv_mod1, 0u);  // unset when crt_count != 2
}

TEST(YpirParamsTest, ParamsForExpansion_Preset) {
  const auto p = ParamsForExpansion(/*nu_1=*/9, /*nu_2=*/6, /*p=*/256,
                                    /*q2_bits=*/28, /*t_exp_left=*/2,
                                    {kQ0, kQ1});
  EXPECT_EQ(p.poly_len, 2048u);
  EXPECT_EQ(p.n, 1u);
  EXPECT_EQ(p.pt_modulus, 256u);
  EXPECT_EQ(p.t_gsw, 3u);
  EXPECT_EQ(p.t_conv, 4u);
  EXPECT_EQ(p.t_exp_left, 2u);
  EXPECT_EQ(p.t_exp_right, 2u);
  EXPECT_EQ(p.instances, 1u);
  EXPECT_EQ(p.db_dim_1, 9u);
  EXPECT_EQ(p.db_dim_2, 6u);
  EXPECT_TRUE(p.expand_queries);
  // db_item_size = 2048 * ceil(log2(256)=8) / 8 = 2048.
  EXPECT_EQ(p.db_item_size, 2048u);
}

TEST(YpirParamsTest, ParamsForExpansion_ClampsQ2Bits) {
  const auto p = ParamsForExpansion(9, 6, 256, /*q2_bits=*/5, 2, {kQ0, kQ1});
  EXPECT_EQ(p.q2_bits, 14u);  // clamped to MIN_Q2_BITS
}

}  // namespace
}  // namespace primihub::pir::ypir
