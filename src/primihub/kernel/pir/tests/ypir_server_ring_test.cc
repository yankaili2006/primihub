/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Tests for ypir_server sub-chunk 10c: YServer::MultiplyWithDbRing, the ring
 * (RLWE) matmul over the transposed DB. Oracle independently recomputes each
 * column's result via a naive negacyclic convolution mod q in Z_q[X]/(X^n+1)
 * (a different algorithm from the NTT path under test), accumulates over the
 * db_dim_1 polynomials, mirrors the SEED_0 negacyclic_perm gate, and applies a
 * manual transpose -- then compares to MultiplyWithDbRing. Covers both the
 * SEED_1 (no perm) and SEED_0 (perm) cases and a non-trivial column slice.
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_util.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

Params MakeParams(std::size_t db_dim_1, std::size_t db_dim_2,
                  std::size_t instances) {
  return Params::Init(8, kModuli, 6.4, 1, 256, 28, 4, 2, 2, 3, true, db_dim_1,
                      db_dim_2, instances, 0, 0);
}

// Naive negacyclic polynomial product in Z_q[X]/(X^n+1): wrap-around terms
// (i+j >= n) are subtracted. Independent of the NTT path under test.
std::vector<std::uint64_t> NaiveNegaConv(const std::vector<std::uint64_t>& a,
                                         const std::vector<std::uint64_t>& b,
                                         std::uint64_t q) {
  const std::size_t n = a.size();
  std::vector<std::uint64_t> out(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const std::uint64_t prod = static_cast<std::uint64_t>(
          (static_cast<__uint128_t>(a[i]) * b[j]) % q);
      const std::size_t k = i + j;
      if (k < n) {
        out[k] = static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(out[k]) + prod) % q);
      } else {
        out[k - n] = static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(out[k - n]) + q - prod) % q);
      }
    }
  }
  return out;
}

// Shared oracle: returns the expected (transposed) MultiplyWithDbRing output.
std::vector<std::uint64_t> ExpectedRing(
    const Params& p, const YServer<std::uint16_t>& srv,
    const std::vector<std::vector<std::uint64_t>>& q_coeffs,
    std::size_t col_start, std::size_t col_end, std::uint8_t seed_idx,
    bool is_simplepir) {
  const std::size_t db_rows_poly = static_cast<std::size_t>(1) << p.db_dim_1;
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::uint16_t* db = srv.Db();
  const std::size_t ncols = col_end - col_start;

  std::vector<std::uint64_t> pre(ncols * p.poly_len, 0);  // ncols x poly_len
  for (std::size_t ci = 0; ci < ncols; ++ci) {
    const std::size_t col = col_start + ci;
    std::vector<std::uint64_t> acc(p.poly_len, 0);
    for (std::size_t r = 0; r < db_rows_poly; ++r) {
      std::vector<std::uint64_t> dbpoly(p.poly_len);
      for (std::size_t z = 0; z < p.poly_len; ++z)
        dbpoly[z] = static_cast<std::uint64_t>(
            db[col * db_rows + r * p.poly_len + z]);
      const std::vector<std::uint64_t> prod =
          NaiveNegaConv(q_coeffs[r], dbpoly, p.modulus);
      for (std::size_t z = 0; z < p.poly_len; ++z)
        acc[z] = static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(acc[z]) + prod[z]) % p.modulus);
    }
    std::vector<std::uint64_t> col_out = acc;
    if (seed_idx == kSeed0 && !is_simplepir)
      col_out = NegacyclicPermU64Mod(acc, 0, p.modulus);
    for (std::size_t z = 0; z < p.poly_len; ++z)
      pre[ci * p.poly_len + z] = col_out[z];
  }

  // transpose ncols x poly_len -> poly_len x ncols
  std::vector<std::uint64_t> expected(p.poly_len * ncols);
  for (std::size_t ci = 0; ci < ncols; ++ci)
    for (std::size_t z = 0; z < p.poly_len; ++z)
      expected[z * ncols + ci] = pre[ci * p.poly_len + z];
  return expected;
}

// Build db_dim_1 query polynomials (1x1 NTT) + return their raw coefficients.
std::vector<PolyMatrixNTT> MakeQuery(
    const NttContext& ctx, const Params& p,
    std::vector<std::vector<std::uint64_t>>* q_coeffs) {
  const std::size_t db_rows_poly = static_cast<std::size_t>(1) << p.db_dim_1;
  std::vector<PolyMatrixNTT> pq;
  q_coeffs->assign(db_rows_poly, std::vector<std::uint64_t>(p.poly_len));
  for (std::size_t r = 0; r < db_rows_poly; ++r) {
    PolyMatrixRaw raw = ctx.ZeroRaw(1, 1);
    for (std::size_t z = 0; z < p.poly_len; ++z) {
      const std::uint64_t v = (r * 101u + z * 7u + 5u) % p.modulus;
      raw.data[z] = v;
      (*q_coeffs)[r][z] = v;
    }
    pq.push_back(ctx.ToNtt(raw));
  }
  return pq;
}

YServer<std::uint16_t> MakeSrv(const Params& p, bool is_simplepir) {
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t cols = DbCols(p, is_simplepir);
  std::vector<std::uint16_t> db(db_rows * cols);
  for (std::size_t k = 0; k < db.size(); ++k)
    db[k] = static_cast<std::uint16_t>((k * 37u + 11u) & 0xFFFFu);
  return YServer<std::uint16_t>(p, db, is_simplepir, false, false);
}

TEST(YpirYServerRingTest, MultiplyWithDbRing_Seed1_MatchesNaiveConv) {
  const Params p = MakeParams(1, 1, 1);
  NttContext ctx(p);
  const YServer<std::uint16_t> srv = MakeSrv(p, /*is_simplepir=*/false);
  std::vector<std::vector<std::uint64_t>> q_coeffs;
  const std::vector<PolyMatrixNTT> pq = MakeQuery(ctx, p, &q_coeffs);

  const std::size_t cs = 1, ce = 4;
  const std::vector<std::uint64_t> out =
      srv.MultiplyWithDbRing(ctx, pq, cs, ce, kSeed1);
  const std::vector<std::uint64_t> expected =
      ExpectedRing(p, srv, q_coeffs, cs, ce, kSeed1, false);

  ASSERT_EQ(out.size(), expected.size());
  EXPECT_EQ(out, expected);
}

TEST(YpirYServerRingTest, MultiplyWithDbRing_Seed0_AppliesNegacyclicPerm) {
  const Params p = MakeParams(1, 1, 1);
  NttContext ctx(p);
  const YServer<std::uint16_t> srv = MakeSrv(p, /*is_simplepir=*/false);
  std::vector<std::vector<std::uint64_t>> q_coeffs;
  const std::vector<PolyMatrixNTT> pq = MakeQuery(ctx, p, &q_coeffs);

  const std::size_t cs = 0, ce = 5;
  const std::vector<std::uint64_t> out =
      srv.MultiplyWithDbRing(ctx, pq, cs, ce, kSeed0);
  const std::vector<std::uint64_t> expected =
      ExpectedRing(p, srv, q_coeffs, cs, ce, kSeed0, false);

  ASSERT_EQ(out.size(), expected.size());
  EXPECT_EQ(out, expected);
}

}  // namespace
}  // namespace primihub::pir::ypir
