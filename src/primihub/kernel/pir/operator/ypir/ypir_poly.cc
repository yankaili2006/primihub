/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"

#include "src/primihub/kernel/pir/operator/ypir/ypir_arith.h"

namespace primihub::pir::ypir {

NttContext::NttContext(const Params& params) : params_(params) {
  for (std::size_t m = 0; m < params_.crt_count && m < kMaxModuli; ++m) {
    ntt_[m] = std::make_unique<intel::hexl::NTT>(params_.poly_len,
                                                 params_.moduli[m]);
  }
}

PolyMatrixRaw NttContext::ZeroRaw(std::size_t rows, std::size_t cols) const {
  PolyMatrixRaw m;
  m.rows = rows;
  m.cols = cols;
  m.data.assign(rows * cols * params_.poly_len, 0);
  return m;
}

PolyMatrixNTT NttContext::ZeroNtt(std::size_t rows, std::size_t cols) const {
  PolyMatrixNTT m;
  m.rows = rows;
  m.cols = cols;
  m.data.assign(rows * cols * params_.crt_count * params_.poly_len, 0);
  return m;
}

PolyMatrixNTT NttContext::ToNtt(const PolyMatrixRaw& in) const {
  const std::size_t pl = params_.poly_len;
  const std::size_t cc = params_.crt_count;
  PolyMatrixNTT out = ZeroNtt(in.rows, in.cols);
  std::vector<std::uint64_t> operand(pl), result(pl);
  for (std::size_t r = 0; r < in.rows; ++r) {
    for (std::size_t c = 0; c < in.cols; ++c) {
      const std::uint64_t* poly = in.Poly(r, c, pl);
      for (std::size_t m = 0; m < cc; ++m) {
        for (std::size_t i = 0; i < pl; ++i) operand[i] = poly[i] % params_.moduli[m];
        ntt_[m]->ComputeForward(result.data(), operand.data(), 1, 1);
        std::uint64_t* dst = out.data.data() + ((r * in.cols + c) * cc + m) * pl;
        for (std::size_t i = 0; i < pl; ++i) dst[i] = result[i];
      }
    }
  }
  return out;
}

PolyMatrixRaw NttContext::FromNtt(const PolyMatrixNTT& in) const {
  const std::size_t pl = params_.poly_len;
  const std::size_t cc = params_.crt_count;
  PolyMatrixRaw out = ZeroRaw(in.rows, in.cols);
  std::vector<std::uint64_t> inv0(pl), inv1(pl), operand(pl);
  for (std::size_t r = 0; r < in.rows; ++r) {
    for (std::size_t c = 0; c < in.cols; ++c) {
      std::uint64_t* dst = out.Poly(r, c, pl);
      const std::uint64_t* base = in.data.data() + (r * in.cols + c) * cc * pl;
      if (cc == 1) {
        for (std::size_t i = 0; i < pl; ++i) operand[i] = base[i];
        ntt_[0]->ComputeInverse(dst, operand.data(), 1, 1);  // modulus == moduli[0]
        continue;
      }
      for (std::size_t i = 0; i < pl; ++i) operand[i] = base[i];
      ntt_[0]->ComputeInverse(inv0.data(), operand.data(), 1, 1);
      for (std::size_t i = 0; i < pl; ++i) operand[i] = base[pl + i];
      ntt_[1]->ComputeInverse(inv1.data(), operand.data(), 1, 1);
      for (std::size_t i = 0; i < pl; ++i) {
        // CRT idempotent reconstruction mod (q0*q1):
        //   x = a0*mod1_inv_mod0 + a1*mod0_inv_mod1  (mod modulus)
        const __uint128_t s =
            static_cast<__uint128_t>(inv0[i]) * params_.mod1_inv_mod0 +
            static_cast<__uint128_t>(inv1[i]) * params_.mod0_inv_mod1;
        dst[i] = BarrettReductionU128Raw(params_.modulus,
                                         params_.barrett_cr_0_modulus,
                                         params_.barrett_cr_1_modulus, s);
      }
    }
  }
  return out;
}

}  // namespace primihub::pir::ypir
