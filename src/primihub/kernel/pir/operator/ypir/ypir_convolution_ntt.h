/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_convolution_ntt — the optimised Convolution struct from upstream
 * src/convolution.rs, chunk 6 pt2. This is the FIRST ypir component to
 * link the C++ Spiral stack's NTT (Intel HEXL, @hexl//:hexl, wired in
 * commit be26cb62 and link-proven by ypir_spiral_smoke_test).
 *
 * Upstream builds a spiral_rs 2-modulus CRT NTT via
 * Convolution::params_for(n) with DEFAULT_MODULI = [268369921,
 * 249561089]. We map that to two intel::hexl::NTT(n, q_m) instances
 * (both moduli are 1 mod 2^16, so HEXL's q == 1 mod 2N holds for any
 * power-of-two n <= 32768). Ntt/Raw use HEXL's negacyclic transform —
 * the intermediate NTT-domain ordering differs from spiral_rs (HEXL's
 * own convention), but Convolve(a,b) is self-consistent and verified
 * against NaiveNegacyclicConvolve.
 *
 * Correctness domain: the product modulus Q = q0*q1 ~ 2^56, so the
 * exact (signed) convolution coefficient must lie in (-Q/2, Q/2] for
 * the u32 result to match the naive u32-wrapping reference. That holds
 * in YPIR's use (one operand is the small gaussian secret key); callers
 * must respect it (it is NOT a general u32 x u32 negacyclic multiply).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_NTT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_NTT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "hexl/ntt/ntt.hpp"

namespace primihub::pir::ypir {

class Convolution {
 public:
  // Builds the two HEXL NTT contexts for transform size n (power of two).
  explicit Convolution(std::size_t n);

  std::size_t n() const { return n_; }

  // CRT parameters of the convolution NTT (the subset of spiral-rs Params that
  // generate_hint_0_ring / answer_hint_ring read via conv.params()).
  // crt_count == 2; poly_len == n; ProductModulus is the product of the two
  // NTT primes; CrtModulus(i) is the i-th prime.
  std::size_t CrtCount() const { return 2; }
  std::size_t PolyLen() const { return n_; }
  std::uint64_t CrtModulus(std::size_t i) const;
  std::uint64_t ProductModulus() const;

  // barrett_coeff_u64(params, val, i): reduce val mod the i-th CRT modulus
  // (single-word Barrett). Folds a lazily-accumulated NTT-domain sum back into
  // a u32 limb before Raw reconstruction.
  std::uint64_t BarrettCoeff(std::uint64_t val, std::size_t i) const;

  // Forward NTT of `a` (length n) under each modulus; returns crt_count*n
  // = 2*n values (modulus m's transform in [m*n, m*n+n)).
  std::vector<std::uint32_t> Ntt(const std::vector<std::uint32_t>& a) const;

  // Inverse NTT under each modulus followed by CRT reconstruction to a
  // signed coefficient mod Q=q0*q1, centred and reduced to u32 (length n).
  std::vector<std::uint32_t> Raw(const std::vector<std::uint32_t>& a) const;

  // Per-modulus pointwise product mod q_m. `a`,`b` are 2*n; returns 2*n.
  std::vector<std::uint32_t> PointwiseMul(
      const std::vector<std::uint32_t>& a,
      const std::vector<std::uint32_t>& b) const;

  // Negacyclic convolution: Raw(PointwiseMul(Ntt(a), Ntt(b))). Length n.
  std::vector<std::uint32_t> Convolve(
      const std::vector<std::uint32_t>& a,
      const std::vector<std::uint32_t>& b) const;

 private:
  std::size_t n_;
  std::unique_ptr<intel::hexl::NTT> ntt_[2];
  std::uint64_t cr1_[2] = {0, 0};  // Barrett cr1 per CRT modulus
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_CONVOLUTION_NTT_H_
