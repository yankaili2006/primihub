/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_server sub-chunk 10b: the YServer<T> data container
 * (new() + transposed DB storage + GetElem/GetRow + smaller_params
 * derivation). Pure data layout -- no HEXL/NTT -- so the oracle replicates
 * the exact fill/index arithmetic of upstream YServer::new and checks the
 * accessors against it, across {transposed, non-transposed} input layouts
 * and {simplepir, doublepir} modes.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params MakeDbParams(std::size_t db_dim_1, std::size_t db_dim_2,
                    std::size_t instances) {
  // poly_len=8, pt_modulus=256, q2_bits=28; t_* / noise irrelevant to YServer.
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, db_dim_1,
                      db_dim_2, instances, 0, 0);
}

// Independent replica of upstream YServer::new fill into a flat T buffer.
template <typename T>
std::vector<T> ExpectedBuf(const Params& p, const std::vector<T>& db,
                           bool is_simplepir, bool inp_transposed,
                           bool pad_rows) {
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t rp = DbRowsPadded(p, pad_rows);
  const std::size_t cols = DbCols(p, is_simplepir);
  std::vector<T> buf(rp * cols, T{});
  std::size_t cnt = 0;
  for (std::size_t i = 0; i < db_rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      buf[inp_transposed ? (i * cols + j) : (j * rp + i)] = db[cnt++];
  return buf;
}

TEST(YpirYServerTest, TransposedStorageRoundTrip) {
  const Params p = MakeDbParams(1, 1, 1);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t cols = DbCols(p, false);
  std::vector<std::uint16_t> db(db_rows * cols);
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint16_t>((k * 7u + 3u) & 0xFFFFu);

  YServer<std::uint16_t> srv(p, db, /*is_simplepir=*/false,
                             /*inp_transposed=*/false, /*pad_rows=*/false);
  EXPECT_EQ(srv.DbColsSelf(), cols);
  EXPECT_EQ(srv.DbRowsPaddedSelf(), DbRowsPadded(p, false));

  const std::vector<std::uint16_t> exp =
      ExpectedBuf<std::uint16_t>(p, db, false, false, false);
  const std::size_t rp = DbRowsPadded(p, false);
  for (std::size_t i = 0; i < db_rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      EXPECT_EQ(srv.GetElem(i, j), exp[j * rp + i]) << i << "," << j;

  for (std::size_t i = 0; i < db_rows; ++i) {
    const std::vector<std::uint16_t> row = srv.GetRow(i);
    ASSERT_EQ(row.size(), cols);
    for (std::size_t j = 0; j < cols; ++j) EXPECT_EQ(row[j], srv.GetElem(i, j));
  }
}

TEST(YpirYServerTest, InpTransposedLayout) {
  const Params p = MakeDbParams(1, 1, 1);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t cols = DbCols(p, false);
  std::vector<std::uint16_t> db(db_rows * cols);
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint16_t>((k * 13u + 1u) & 0xFFFFu);

  YServer<std::uint16_t> srv(p, db, false, /*inp_transposed=*/true, false);
  const std::vector<std::uint16_t> exp =
      ExpectedBuf<std::uint16_t>(p, db, false, true, false);
  const std::size_t rp = DbRowsPadded(p, false);
  for (std::size_t i = 0; i < db_rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      EXPECT_EQ(srv.GetElem(i, j), exp[j * rp + i]) << i << "," << j;
}

TEST(YpirYServerTest, SimplePirDbColsAndSmallerParamsCopy) {
  const Params p = MakeDbParams(1, 1, 3);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t cols = DbCols(p, true);
  EXPECT_EQ(cols, p.instances * p.poly_len);  // 3 * 8 = 24

  std::vector<std::uint8_t> db(db_rows * cols);
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint8_t>(k * 5u + 2u);

  YServer<std::uint8_t> srv(p, db, /*is_simplepir=*/true, false, false);
  EXPECT_EQ(srv.DbColsSelf(), cols);
  // simplepir: smaller_params is an unchanged copy of params
  EXPECT_EQ(srv.smaller_params().db_dim_1, p.db_dim_1);
  EXPECT_EQ(srv.smaller_params().db_dim_2, p.db_dim_2);

  const std::vector<std::uint8_t> exp =
      ExpectedBuf<std::uint8_t>(p, db, true, false, false);
  const std::size_t rp = DbRowsPadded(p, false);
  for (std::size_t i = 0; i < db_rows; ++i)
    for (std::size_t j = 0; j < cols; ++j)
      EXPECT_EQ(srv.GetElem(i, j), exp[j * rp + i]) << i << "," << j;
}

TEST(YpirYServerTest, SmallerParamsDoublePirDerivation) {
  const Params p = MakeDbParams(1, 2, 1);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  std::vector<std::uint16_t> db(db_rows * DbCols(p, false));
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint16_t>(k & 0xFFFFu);

  YServer<std::uint16_t> srv(p, db, /*is_simplepir=*/false, false, false);
  // smaller.db_dim_1 = params.db_dim_2
  EXPECT_EQ(srv.smaller_params().db_dim_1, p.db_dim_2);
  // smaller.db_dim_2 = ceil(log2(blowup * (n+1) / poly_len)),
  // blowup = q2_bits/pt_bits = 28/8 = 3.5, n=1024, poly_len=8
  //   -> ceil(log2(3.5 * 1025 / 8)) = ceil(8.808...) = 9
  EXPECT_EQ(srv.smaller_params().db_dim_2, static_cast<std::size_t>(9));
}

}  // namespace
}  // namespace primihub::pir::ypir
