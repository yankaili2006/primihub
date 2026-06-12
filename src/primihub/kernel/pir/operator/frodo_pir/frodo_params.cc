/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_params.h"

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

std::vector<std::vector<std::uint32_t>> BaseParams::GenerateParamsRhs(
    const Database& db, const SeedBytes& public_seed,
    std::size_t dim, std::size_t m) {
  // Upstream:
  //   let lhs = swap_matrix_fmt(
  //       &generate_lwe_matrix_from_seed(public_seed, dim, m));
  //   (0..db.get_matrix_width_self())
  //     .map(|i| {
  //       let mut col = Vec::with_capacity(m);
  //       for r in &lhs { col.push(db.vec_mult(r, i)); }
  //       col
  //     })
  //     .collect()
  //
  // Note: upstream's `col` has Vec::with_capacity(m) but actually
  // ends up with `lhs.len() == dim` entries because the inner
  // loop iterates over `lhs`'s columns. We match that — RHS column
  // length is `dim`, not `m`. This is upstream code as written;
  // the `with_capacity(m)` is a minor upstream copy-paste artifact
  // since on the working FrodoPIR parameter sets m == dim is
  // unusual but the buffer grows on first push regardless.
  const auto lhs_seed =
      GenerateLweMatrixFromSeed(public_seed, dim, m);
  const auto lhs = SwapMatrixFmt(lhs_seed);
  // lhs has `dim` columns of `m` u32s each.

  const std::size_t w = db.GetMatrixWidthSelf();
  std::vector<std::vector<std::uint32_t>> rhs;
  rhs.reserve(w);
  for (std::size_t i = 0; i < w; ++i) {
    std::vector<std::uint32_t> col;
    col.reserve(lhs.size());
    for (const auto& r : lhs) {
      col.push_back(db.VecMult(r, i));
    }
    rhs.push_back(std::move(col));
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
  out->rhs_ = GenerateParamsRhs(db, seed, dim, out->m_);
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
  // Upstream:
  //   (0..cols.len()).map(|i| vec_mult_u32_u32(s, &cols[i])).collect()
  std::vector<std::uint32_t> result;
  result.reserve(rhs_.size());
  for (std::size_t i = 0; i < rhs_.size(); ++i) {
    std::uint32_t v = 0;
    const auto rc = VecMultU32U32(s, rhs_[i], &v, err);
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

// ---------- CommonParams ----------

CommonParams::CommonParams() = default;

CommonParams::CommonParams(
    std::vector<std::vector<std::uint32_t>> matrix)
    : matrix_(std::move(matrix)) {}

CommonParams CommonParams::FromBaseParams(const BaseParams& params) {
  // Upstream: Self(generate_lwe_matrix_from_seed(
  //     params.public_seed, params.dim, params.m))
  return CommonParams(GenerateLweMatrixFromSeed(
      params.GetPublicSeed(), params.GetDim(),
      params.GetTotalRecords()));
}

retcode CommonParams::MultLeft(
    const std::vector<std::uint32_t>& s,
    std::vector<std::uint32_t>* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "CommonParams::MultLeft: out must be non-null";
    return retcode::FAIL;
  }
  // Upstream:
  //   (0..cols.len()).map(|i| {
  //     let s_a = vec_mult_u32_u32(s, &cols[i])?;
  //     let e = random_ternary();
  //     Ok(s_a.wrapping_add(e))
  //   }).collect()
  std::vector<std::uint32_t> result;
  result.reserve(matrix_.size());
  for (std::size_t i = 0; i < matrix_.size(); ++i) {
    std::uint32_t s_a = 0;
    const auto rc = VecMultU32U32(s, matrix_[i], &s_a, err);
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
