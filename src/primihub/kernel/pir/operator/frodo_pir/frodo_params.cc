/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"

#include <algorithm>
#include <sstream>
#include <utility>

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

namespace primihub::pir::frodo {

BaseParams::BaseParams()
    : dim_(0),
      m_(0),
      elem_size_(0),
      plaintext_bits_(0),
      public_seed_{} {}

ColMajorMatrix BaseParams::GenerateParamsRhsFlat(
    const Database& db, const SeedBytes& public_seed,
    std::size_t dim, std::size_t m) {
  // chunk g-5: flat-form GenerateParamsRhs. Mirrors upstream
  //   let lhs = swap_matrix_fmt(
  //       &generate_lwe_matrix_from_seed(public_seed, dim, m));
  //   rhs[i] = (0..lhs.len()).map(|j| db.vec_mult(lhs[j], i))
  //   for i in 0..db.get_matrix_width_self()
  //
  // Step 1: GenerateLweMatrixFromSeedFlat produces an `m x dim`
  // column-major matrix (height=dim, width=m, total dim*m u32s).
  // Step 2: SwapMatrixFmtFlat swaps to `dim x m` (height=m,
  // width=dim) so each column j is a contiguous run of `m` u32s.
  // Step 3: For each db column i in [0, w), and each lhs column
  // j in [0, dim), compute rhs.at(i, j) = wrapping dot-product
  // of lhs.column j against db's i-th column. The chunk g-4
  // raw-pointer VecMultU32U32 overload reads both column streams
  // directly out of their flat backings -- no per-call vector
  // temporary.
  const auto lhs_seed_flat =
      GenerateLweMatrixFromSeedFlat(public_seed, dim, m);
  const auto lhs = SwapMatrixFmtFlat(lhs_seed_flat);
  // lhs shape: height=m, width=dim.

  const std::size_t w = db.GetMatrixWidthSelf();
  ColMajorMatrix rhs(/*height=*/dim, /*width=*/w,
                     ColMajorMatrix::NoInit{});
  // chunk g-5 follow-up: feed both lhs columns and db columns to
  // the raw-pointer VecMult path. The prior implementation
  // constructed a vector<uint32_t>(m) per inner iteration, paying
  // w*dim allocations of m u32s each (2 GB / 500 K page faults
  // at N=1M / dim=512). Now the inner loop touches only the
  // SwapMatrixFmtFlat-produced lhs storage and the Database flat
  // entries; no per-iteration heap traffic.
  for (std::size_t i = 0; i < w; ++i) {
    std::uint32_t* rhs_col = rhs.column_data(i);
    for (std::size_t j = 0; j < dim; ++j) {
      const std::uint32_t* lhs_col = lhs.column_data(j);
      rhs_col[j] = db.VecMult(lhs_col, m, i);
    }
  }
  return rhs;
}

std::vector<std::vector<std::uint32_t>> BaseParams::GenerateParamsRhs(
    const Database& db, const SeedBytes& public_seed,
    std::size_t dim, std::size_t m) {
  // chunk g-5: nested form delegates to the flat helper and
  // materialises the nested vector at the boundary. Result shape:
  // outer size = w, inner size = dim, byte-for-byte equivalent to
  // the prior `vector<vector<u32>>` implementation.
  const auto flat = GenerateParamsRhsFlat(db, public_seed, dim, m);
  const std::size_t w = flat.width();
  const std::size_t h = flat.height();
  std::vector<std::vector<std::uint32_t>> rhs;
  rhs.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    const std::uint32_t* col = flat.column_data(i);
    rhs.emplace_back(col, col + h);
  }
  return rhs;
}

retcode BaseParams::NewWithSeed(const Database& db, std::size_t dim,
                                const SeedBytes& seed, BaseParams* out,
                                std::string* err) {
  if (out == nullptr) {
    if (err) *err = "BaseParams::NewWithSeed: out must be non-null";
    return retcode::FAIL;
  }
  out->dim_ = dim;
  out->m_ = db.GetMatrixHeight();
  out->elem_size_ = db.GetElemSize();
  out->plaintext_bits_ = db.GetPlaintextBits();
  out->public_seed_ = seed;
  // chunk g-5: route production through the flat helper. Avoids
  // the nested-form materialisation that the test-facing
  // GenerateParamsRhs wrapper still performs.
  out->rhs_ = GenerateParamsRhsFlat(db, seed, dim, out->m_);
  return retcode::SUCCESS;
}

retcode BaseParams::New(const Database& db, std::size_t dim,
                        BaseParams* out, std::string* err) {
  return NewWithSeed(db, dim, GenerateSeed(), out, err);
}

retcode BaseParams::MultRight(
    const std::vector<std::uint32_t>& s,
    std::vector<std::uint32_t>* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "BaseParams::MultRight: out must be non-null";
    return retcode::FAIL;
  }
  // chunk g-5: iterate rhs_ via column_data + height instead of
  // entries_[i] -- no per-call vector temporary, no dangling
  // reference risk. Otherwise identical to the prior
  // upstream-mirrored loop:
  //   (0..cols.len()).map(|i| vec_mult_u32_u32(s, &cols[i])).collect()
  const std::size_t w = rhs_.width();
  const std::size_t h = rhs_.height();
  std::vector<std::uint32_t> result;
  result.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    std::uint32_t v = 0;
    const auto rc = VecMultU32U32(s, rhs_.column_data(i), h, &v, err);
    if (rc != retcode::SUCCESS) {
      if (err) {
        std::ostringstream oss;
        oss << "BaseParams::MultRight: rhs column " << i
            << " dot product failed: " << *err;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    result.push_back(v);
  }
  *out = std::move(result);
  return retcode::SUCCESS;
}

std::vector<std::vector<std::uint32_t>> BaseParams::RhsForTest() const {
  // chunk g-5: materialise the column-major flat storage into the
  // legacy nested form (one inner vec per column, height u32s
  // each). Test-only path; production reads rhs_ directly via
  // RhsFlat() or MultRight()'s column_data path.
  const std::size_t w = rhs_.width();
  const std::size_t h = rhs_.height();
  std::vector<std::vector<std::uint32_t>> out;
  out.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    const std::uint32_t* col = rhs_.column_data(i);
    out.emplace_back(col, col + h);
  }
  return out;
}

// ---------- CommonParams ----------

CommonParams::CommonParams() = default;

CommonParams::CommonParams(
    std::vector<std::vector<std::uint32_t>> matrix) {
  // chunk g-5: nested-form ctor (tests construct CommonParams
  // directly with hand-built vec<vec<u32>> fixtures). Copy into a
  // ColMajorMatrix of identical shape (height = matrix[0].size(),
  // width = matrix.size()) so MultLeft and AsMatrix() both see
  // the flat backing from here on.
  const std::size_t width = matrix.size();
  const std::size_t height = width == 0 ? 0 : matrix[0].size();
  matrix_ = ColMajorMatrix(height, width, ColMajorMatrix::NoInit{});
  for (std::size_t c = 0; c < width; ++c) {
    std::uint32_t* dst = matrix_.column_data(c);
    const std::vector<std::uint32_t>& src = matrix[c];
    const std::size_t n = std::min(height, src.size());
    for (std::size_t r = 0; r < n; ++r) {
      dst[r] = src[r];
    }
  }
}

CommonParams::CommonParams(ColMajorMatrix matrix)
    : matrix_(std::move(matrix)) {}

CommonParams CommonParams::FromBaseParams(const BaseParams& params) {
  // chunk g-5: flat path. GenerateLweMatrixFromSeedFlat returns a
  // ColMajorMatrix; the flat-form ctor takes it by move so no copy
  // happens at the construction boundary.
  return CommonParams(GenerateLweMatrixFromSeedFlat(
      params.GetPublicSeed(), params.GetDim(),
      params.GetTotalRecords()));
}

std::vector<std::vector<std::uint32_t>> CommonParams::AsMatrix() const {
  // chunk g-5: materialise the flat storage into the legacy nested
  // form for tests that compare against vec<vec<u32>> literals or
  // pass `AsMatrix()[i]` to vector-taking helpers.
  const std::size_t w = matrix_.width();
  const std::size_t h = matrix_.height();
  std::vector<std::vector<std::uint32_t>> out;
  out.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    const std::uint32_t* col = matrix_.column_data(i);
    out.emplace_back(col, col + h);
  }
  return out;
}

retcode CommonParams::MultLeft(
    const std::vector<std::uint32_t>& s,
    std::vector<std::uint32_t>* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "CommonParams::MultLeft: out must be non-null";
    return retcode::FAIL;
  }
  // chunk g-5: iterate matrix_ via column_data + height. Upstream
  // semantics preserved:
  //   (0..cols.len()).map(|i| {
  //     let s_a = vec_mult_u32_u32(s, &cols[i])?;
  //     let e = random_ternary();
  //     Ok(s_a.wrapping_add(e))
  //   }).collect()
  const std::size_t w = matrix_.width();
  const std::size_t h = matrix_.height();
  std::vector<std::uint32_t> result;
  result.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    std::uint32_t s_a = 0;
    const auto rc =
        VecMultU32U32(s, matrix_.column_data(i), h, &s_a, err);
    if (rc != retcode::SUCCESS) {
      if (err) {
        std::ostringstream oss;
        oss << "CommonParams::MultLeft: matrix column " << i
            << " dot product failed: " << *err;
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    const std::uint32_t e = RandomTernary();
    // C++ unsigned wraps mod 2^32 by spec — matches upstream
    // wrapping_add.
    result.push_back(s_a + e);
  }
  *out = std::move(result);
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::frodo
