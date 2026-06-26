/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for YServer::GenerateHint0Ring (server.rs generate_hint_0_ring) plus
 * the Log2 helper. The hint oracle independently recomputes H0 = A * DB via a
 * naive negacyclic convolution (NaiveNegacyclicConvolve) summed over the
 * db_rows/n blocks (mod 2^32) -- a different algorithm from the lazily-
 * accumulated CRT-NTT path under test. db values are kept < pt_modulus so the
 * no-Q-overflow / max_adds analysis holds.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_negacyclic.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_scheme.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params MakeParams(std::size_t db_dim_1, std::size_t db_dim_2,
                  std::size_t instances) {
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, db_dim_1,
                      db_dim_2, instances, 0, 0);
}

TEST(YpirServerHintTest, Log2BasicValues) {
  EXPECT_EQ(Log2(0), 0u);
  EXPECT_EQ(Log2(1), 0u);
  EXPECT_EQ(Log2(2), 1u);
  EXPECT_EQ(Log2(3), 1u);
  EXPECT_EQ(Log2(255), 7u);
  EXPECT_EQ(Log2(256), 8u);
  EXPECT_EQ(Log2(static_cast<std::uint64_t>(1) << 32), 32u);
}

TEST(YpirServerHintTest, GenerateHint0RingMatchesNaiveConv) {
  const std::size_t n = 1024;  // LWE n
  // db_rows = 2^(8+3) = 2048 (2 outer rows over n=1024); db_cols = 2^(0+3) = 8.
  const Params p = MakeParams(8, 0, 1);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t db_cols = DbCols(p, false);
  ASSERT_EQ(db_rows, 2048u);
  ASSERT_EQ(db_cols, 8u);

  std::vector<std::uint16_t> db(db_rows * db_cols);
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint16_t>((k * 37u + 11u) % 256u);  // < pt_modulus

  const YServer<std::uint16_t> srv(p, db, /*is_simplepir=*/false,
                                   /*inp_transposed=*/false, /*pad_rows=*/false);

  const std::vector<std::uint64_t> hint = srv.GenerateHint0Ring();
  ASSERT_EQ(hint.size(), n * db_cols);

  // Oracle: A is block-negacyclic from get_seed(SEED_0); H0[i][col] =
  // sum_blocks NaiveNegacyclicConvolve(nega_perm_a[block], db_col_block)[i].
  const std::size_t num_outer = db_rows / n;
  const std::size_t rp = DbRowsPadded(p, false);
  ChaChaRng rng = ChaChaRng::FromSeed(GetSeed(0));
  std::vector<std::vector<std::uint32_t>> nega(num_outer);
  for (std::size_t outer = 0; outer < num_outer; ++outer) {
    std::vector<std::uint32_t> a(n);
    for (std::size_t idx = 0; idx < n; ++idx) a[idx] = rng.NextU32();
    nega[outer] = NegacyclicPermU32(a);
  }

  const std::uint16_t* dbp = srv.Db();
  std::vector<std::uint64_t> expected(n * db_cols, 0);
  for (std::size_t col = 0; col < db_cols; ++col) {
    std::vector<std::uint64_t> acc(n, 0);
    for (std::size_t outer = 0; outer < num_outer; ++outer) {
      std::vector<std::uint32_t> pt(n);
      for (std::size_t z = 0; z < n; ++z)
        pt[z] = static_cast<std::uint32_t>(dbp[col * rp + outer * n + z]);
      const std::vector<std::uint32_t> cv =
          NaiveNegacyclicConvolve(nega[outer], pt);
      for (std::size_t i = 0; i < n; ++i)
        acc[i] = (acc[i] + cv[i]) & 0xFFFFFFFFull;
    }
    for (std::size_t i = 0; i < n; ++i) expected[i * db_cols + col] = acc[i];
  }

  EXPECT_EQ(hint, expected);
}

}  // namespace
}  // namespace primihub::pir::ypir
