/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_bits.h"

#include <algorithm>

namespace primihub::pir::ypir {

void WriteBits(std::uint8_t* data, std::size_t data_len,
               std::uint64_t val, std::size_t bit_offs,
               std::size_t num_bits) {
  if (data == nullptr || data_len == 0 || num_bits == 0) {
    return;
  }
  std::size_t byte_index = bit_offs / 8u;
  std::size_t bit_index = bit_offs % 8u;
  while (num_bits > 0 && byte_index < data_len) {
    // bits_to_write = min(8 - bit_index, num_bits)
    std::size_t bits_to_write = std::min(8u - static_cast<std::uint32_t>(bit_index),
                                          static_cast<std::uint32_t>(num_bits));
    // Mask out the low bits_to_write bits of val. `(1 << bits_to_write) - 1`
    // is well-defined for bits_to_write in [1, 64].
    std::uint64_t bitmask =
        (bits_to_write == 64u)
            ? ~static_cast<std::uint64_t>(0)
            : ((static_cast<std::uint64_t>(1) << bits_to_write) - 1u);
    std::uint64_t bits = (val & bitmask) << bit_index;
    data[byte_index] = static_cast<std::uint8_t>(
        static_cast<std::uint64_t>(data[byte_index]) | bits);
    num_bits -= bits_to_write;
    bit_index += bits_to_write;
    if (bit_index == 8u) {
      ++byte_index;
      bit_index = 0;
    }
    val >>= bits_to_write;
  }
}

std::uint64_t ReadBits(const std::uint8_t* data, std::size_t data_len,
                       std::size_t bit_offs, std::size_t num_bits) {
  if (data == nullptr || num_bits == 0 || num_bits > 64) {
    return 0u;
  }
  std::size_t byte_pos = bit_offs / 8u;
  std::size_t bit_pos = bit_offs % 8u;
  std::uint64_t result = 0u;
  std::size_t remaining_bits = num_bits;
  for (std::size_t i = byte_pos; i < data_len; ++i) {
    std::size_t can_take = std::min(8u - static_cast<std::uint32_t>(bit_pos),
                                     static_cast<std::uint32_t>(remaining_bits));
    std::uint8_t value;
    if (can_take < 8) {
      value = static_cast<std::uint8_t>(
          (data[i] >> bit_pos) & ((1u << can_take) - 1u));
    } else {
      value = static_cast<std::uint8_t>(data[i] >> bit_pos);
    }
    result |= static_cast<std::uint64_t>(value) << (num_bits - remaining_bits);
    remaining_bits -= can_take;
    if (remaining_bits == 0) {
      break;
    }
    // Upstream branch: if we didn't consume a whole byte and there
    // are remaining bits, take the lower bits of THIS byte and the
    // higher from next iteration. Mirrors upstream byte-for-byte.
    if (bit_pos + can_take < 8u) {
      std::uint8_t from_next_byte = static_cast<std::uint8_t>(
          data[i] & ((1u << (bit_pos + can_take)) - 1u));
      result |= static_cast<std::uint64_t>(from_next_byte)
                << (num_bits - remaining_bits);
      remaining_bits -= bit_pos + can_take;
    }
    bit_pos = 0;  // subsequent bytes start from the LSB
  }
  return result;
}

std::vector<std::uint8_t> U64sToContiguousBytes(
    const std::vector<std::uint64_t>& data, std::size_t inp_mod_bits) {
  if (inp_mod_bits == 0 || inp_mod_bits > 64) {
    return {};
  }
  // total_sz = ceil(data.len() * inp_mod_bits / 8)
  const std::size_t total_bits = data.size() * inp_mod_bits;
  const std::size_t total_sz = (total_bits + 7u) / 8u;
  std::vector<std::uint8_t> out(total_sz, 0u);
  std::size_t bit_offs = 0;
  for (std::size_t i = 0; i < data.size(); ++i) {
    WriteBits(out.data(), out.size(), data[i], bit_offs, inp_mod_bits);
    bit_offs += inp_mod_bits;
  }
  return out;
}

std::vector<std::uint64_t> ContiguousBytesToU64s(
    const std::vector<std::uint8_t>& data, std::size_t out_mod_bits) {
  if (out_mod_bits == 0 || out_mod_bits > 64) {
    return {};
  }
  const std::size_t n_out = (data.size() * 8u) / out_mod_bits;
  std::vector<std::uint64_t> out(n_out, 0u);
  std::size_t bit_offs = 0;
  for (std::size_t i = 0; i < n_out; ++i) {
    out[i] = ReadBits(data.data(), data.size(), bit_offs, out_mod_bits);
    bit_offs += out_mod_bits;
  }
  return out;
}

}  // namespace primihub::pir::ypir
