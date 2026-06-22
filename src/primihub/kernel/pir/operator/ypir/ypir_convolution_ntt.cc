/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_convolution_ntt.h"

#include <cstdint>

namespace primihub::pir::ypir {

namespace {

// ypir DEFAULT_MODULI (src/params.rs): two NTT primes, both 1 mod 2^16.
constexpr std::uint64_t kQ[2] = {268369921ull, 249561089ull};

// a^{-1} mod m via the extended Euclidean algorithm (a, m < 2^28).
std::uint64_t ModInverse(std::uint64_t a, std::uint64_t m) {
  std::int64_t t = 0, new_t = 1;
  std::int64_t r = static_cast<std::int64_t>(m);
  std::int64_t new_r = static_cast<std::int64_t>(a);
  while (new_r != 0) {
    const std::int64_t q = r / new_r;
    std::int64_t tmp = t - q * new_t;
    t = new_t;
    new_t = tmp;
    tmp = r - q * new_r;
    r = new_r;
    new_r = tmp;
  }
  if (t < 0) t += static_cast<std::int64_t>(m);
  return static_cast<std::uint64_t>(t);
}

}  // namespace

Convolution::Convolution(std::size_t n) : n_(n) {
  ntt_[0] = std::make_unique<intel::hexl::NTT>(n, kQ[0]);
  ntt_[1] = std::make_unique<intel::hexl::NTT>(n, kQ[1]);
}

std::vector<std::uint32_t> Convolution::Ntt(
    const std::vector<std::uint32_t>& a) const {
  std::vector<std::uint32_t> out(2 * n_);
  std::vector<std::uint64_t> operand(n_), result(n_);
  for (int m = 0; m < 2; ++m) {
    for (std::size_t i = 0; i < n_; ++i) {
      operand[i] = static_cast<std::uint64_t>(a[i]) % kQ[m];
    }
    ntt_[m]->ComputeForward(result.data(), operand.data(), 1, 1);
    for (std::size_t i = 0; i < n_; ++i) {
      out[m * n_ + i] = static_cast<std::uint32_t>(result[i]);
    }
  }
  return out;
}

std::vector<std::uint32_t> Convolution::PointwiseMul(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b) const {
  std::vector<std::uint32_t> out(2 * n_);
  for (int m = 0; m < 2; ++m) {
    for (std::size_t i = 0; i < n_; ++i) {
      const std::size_t idx = m * n_ + i;
      const std::uint64_t v =
          (static_cast<std::uint64_t>(a[idx]) * b[idx]) % kQ[m];
      out[idx] = static_cast<std::uint32_t>(v);
    }
  }
  return out;
}

std::vector<std::uint32_t> Convolution::Raw(
    const std::vector<std::uint32_t>& a) const {
  // Inverse NTT under each modulus.
  std::vector<std::uint64_t> r[2] = {std::vector<std::uint64_t>(n_),
                                     std::vector<std::uint64_t>(n_)};
  std::vector<std::uint64_t> operand(n_);
  for (int m = 0; m < 2; ++m) {
    for (std::size_t i = 0; i < n_; ++i) operand[i] = a[m * n_ + i];
    ntt_[m]->ComputeInverse(r[m].data(), operand.data(), 1, 1);
  }

  // CRT-combine (Garner) + centre to (-Q/2, Q/2] + reduce to u32.
  const __uint128_t Q = static_cast<__uint128_t>(kQ[0]) * kQ[1];
  const std::uint64_t inv_q0_mod_q1 = ModInverse(kQ[0] % kQ[1], kQ[1]);

  std::vector<std::uint32_t> out(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    const std::uint64_t a0 = r[0][i];               // V mod q0
    const std::uint64_t a1 = r[1][i];               // V mod q1
    // t = (a1 - a0) * q0^{-1} mod q1
    const std::uint64_t diff = (a1 + kQ[1] - (a0 % kQ[1])) % kQ[1];
    const std::uint64_t t =
        static_cast<std::uint64_t>(
            (static_cast<__uint128_t>(diff) * inv_q0_mod_q1) % kQ[1]);
    const __uint128_t V =
        static_cast<__uint128_t>(a0) + static_cast<__uint128_t>(kQ[0]) * t;
    // Centre: signed representative in (-Q/2, Q/2].
    __int128_t Vs = static_cast<__int128_t>(V);
    if (V > Q / 2) Vs -= static_cast<__int128_t>(Q);
    // Reduce mod 2^32 (two's-complement low word handles the sign).
    out[i] = static_cast<std::uint32_t>(static_cast<std::int64_t>(Vs));
  }
  return out;
}

std::vector<std::uint32_t> Convolution::Convolve(
    const std::vector<std::uint32_t>& a,
    const std::vector<std::uint32_t>& b) const {
  return Raw(PointwiseMul(Ntt(a), Ntt(b)));
}

}  // namespace primihub::pir::ypir
