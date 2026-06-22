/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"

#include <cmath>

#include "src/primihub/kernel/pir/operator/ypir/ypir_bits.h"

namespace primihub::pir::ypir {

namespace {

// ceil(log2(x)), matching upstream `(x as f64).log2().ceil() as usize`.
std::size_t CeilLog2(std::uint64_t x) {
  return static_cast<std::size_t>(
      std::ceil(std::log2(static_cast<double>(x))));
}

}  // namespace

std::uint64_t Rescale(std::uint64_t a, std::uint64_t inp_mod,
                      std::uint64_t out_mod) {
  // Mirror of menonsamir/spiral src/poly.cpp:rescale (nosign=false).
  std::int64_t inp_val = static_cast<std::int64_t>(a % inp_mod);
  if (inp_val >= static_cast<std::int64_t>(inp_mod / 2)) {
    inp_val -= static_cast<std::int64_t>(inp_mod);
  }
  const std::int64_t sign = inp_val >= 0 ? 1 : -1;
  const __int128 val =
      static_cast<__int128>(inp_val) * static_cast<__int128>(out_mod);
  __int128 result =
      (val + sign * static_cast<std::int64_t>(inp_mod / 2)) /
      static_cast<__int128>(inp_mod);
  result = (result + static_cast<__int128>(inp_mod / out_mod) * out_mod +
            2 * static_cast<__int128>(out_mod)) %
           out_mod;
  return static_cast<std::uint64_t>((result + out_mod) % out_mod);
}

std::vector<std::uint8_t> ModulusSwitchPack(
    const std::uint64_t* row0, const std::uint64_t* row1,
    std::size_t poly_len, std::uint64_t modulus,
    std::uint64_t q_1, std::uint64_t q_2) {
  const std::size_t q_1_bits = CeilLog2(q_2);
  const std::size_t q_2_bits = CeilLog2(q_1);
  const std::size_t total_sz_bits = (q_1_bits + q_2_bits) * poly_len;
  const std::size_t total_sz_bytes = (total_sz_bits + 7) / 8;

  std::vector<std::uint8_t> res(total_sz_bytes, 0);
  std::size_t bit_offs = 0;
  for (std::size_t z = 0; z < poly_len; ++z) {
    const std::uint64_t v = Rescale(row0[z], modulus, q_2);
    WriteBits(res.data(), res.size(), v, bit_offs, q_1_bits);
    bit_offs += q_1_bits;
  }
  for (std::size_t z = 0; z < poly_len; ++z) {
    const std::uint64_t v = Rescale(row1[z], modulus, q_1);
    WriteBits(res.data(), res.size(), v, bit_offs, q_2_bits);
    bit_offs += q_2_bits;
  }
  return res;
}

bool ModulusSwitchRecover(
    const std::vector<std::uint8_t>& ciphertext,
    std::size_t poly_len, std::uint64_t modulus,
    std::uint64_t q_1, std::uint64_t q_2,
    std::vector<std::uint64_t>* row0,
    std::vector<std::uint64_t>* row1) {
  if (row0 == nullptr || row1 == nullptr) return false;
  const std::size_t q_1_bits = CeilLog2(q_2);
  const std::size_t q_2_bits = CeilLog2(q_1);
  const std::size_t total_sz_bits = (q_1_bits + q_2_bits) * poly_len;
  const std::size_t total_sz_bytes = (total_sz_bits + 7) / 8;
  if (ciphertext.size() != total_sz_bytes) return false;

  row0->assign(poly_len, 0);
  row1->assign(poly_len, 0);
  std::size_t bit_offs = 0;
  for (std::size_t z = 0; z < poly_len; ++z) {
    const std::uint64_t v =
        ReadBits(ciphertext.data(), ciphertext.size(), bit_offs, q_1_bits);
    (*row0)[z] = Rescale(v, q_2, modulus);
    bit_offs += q_1_bits;
  }
  for (std::size_t z = 0; z < poly_len; ++z) {
    const std::uint64_t v =
        ReadBits(ciphertext.data(), ciphertext.size(), bit_offs, q_2_bits);
    (*row1)[z] = Rescale(v, q_1, modulus);
    bit_offs += q_2_bits;
  }
  return true;
}

}  // namespace primihub::pir::ypir
