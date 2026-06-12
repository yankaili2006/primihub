/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_format.h"

#include <sstream>

namespace primihub::pir::frodo {

std::vector<std::uint8_t> U8ToBitsLe(std::uint8_t byte) {
  std::vector<std::uint8_t> ret(8, 0u);
  for (std::size_t i = 0; i < 8; ++i) {
    // 2^i & byte > 0 -> 1 else 0
    ret[i] = (((static_cast<std::uint32_t>(1) << i) & byte) > 0u) ? 1u : 0u;
  }
  return ret;
}

std::vector<std::uint8_t> U32ToBitsLe(std::uint32_t x,
                                      std::size_t bit_len) {
  // Upstream extracts the 4 little-endian bytes of x, expands each
  // to 8 bits, then truncates to bit_len. We match exactly.
  std::vector<std::uint8_t> bits;
  bits.reserve(32);
  for (std::size_t i = 0; i < 4; ++i) {
    std::uint8_t byte = static_cast<std::uint8_t>((x >> (i * 8)) & 0xFFu);
    auto byte_bits = U8ToBitsLe(byte);
    bits.insert(bits.end(), byte_bits.begin(), byte_bits.end());
  }
  if (bit_len > bits.size()) {
    bit_len = bits.size();  // clamp instead of panic
  }
  bits.resize(bit_len);
  return bits;
}

std::vector<std::uint8_t> BitsToBytesLe(
    const std::vector<std::uint8_t>& bits) {
  const std::size_t out_sz = (bits.size() + 7) / 8u;
  std::vector<std::uint8_t> bytes(out_sz, 0u);
  for (std::size_t i = 0; i < bits.size(); ++i) {
    if (bits[i] != 0) {
      const std::size_t idx = i / 8u;
      const std::uint32_t exp = static_cast<std::uint32_t>(i % 8u);
      bytes[idx] = static_cast<std::uint8_t>(
          bytes[idx] + (static_cast<std::uint32_t>(1) << exp));
    }
  }
  return bytes;
}

std::vector<std::uint8_t> BytesToBitsLe(
    const std::vector<std::uint8_t>& bytes) {
  std::vector<std::uint8_t> out;
  out.reserve(bytes.size() * 8u);
  for (std::uint8_t b : bytes) {
    auto bits = U8ToBitsLe(b);
    out.insert(out.end(), bits.begin(), bits.end());
  }
  return out;
}

retcode BitsToU32Le(const std::vector<std::uint8_t>& bits,
                    std::uint32_t* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "BitsToU32Le: out must be non-null";
    return retcode::FAIL;
  }
  auto bytes = BitsToBytesLe(bits);
  constexpr std::size_t kU32Len = sizeof(std::uint32_t);
  if (bytes.size() > kU32Len) {
    if (err) {
      std::ostringstream oss;
      oss << "BitsToU32Le: bytes too long to parse as u32, "
          << "length: " << bytes.size()
          << " (max " << kU32Len << "). Upstream raises "
          << "ErrorUnexpectedInputSize here.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Pad with zeros to exactly 4 bytes, then read little-endian.
  while (bytes.size() < kU32Len) {
    bytes.push_back(0);
  }
  std::uint32_t v = 0;
  for (std::size_t i = 0; i < kU32Len; ++i) {
    v |= static_cast<std::uint32_t>(bytes[i]) << (i * 8u);
  }
  *out = v;
  return retcode::SUCCESS;
}

std::vector<std::uint8_t> BytesFromU32Slice(
    const std::vector<std::uint32_t>& v, std::size_t entry_bit_len,
    std::size_t total_bit_len) {
  if (v.empty() || entry_bit_len == 0) {
    return {};
  }
  const std::size_t remainder = total_bit_len % entry_bit_len;
  std::vector<std::uint8_t> bits;
  bits.reserve(entry_bit_len * v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i + 1 != v.size()) {
      auto chunk = U32ToBitsLe(v[i], entry_bit_len);
      bits.insert(bits.end(), chunk.begin(), chunk.end());
    } else {
      // Upstream behavior: the last entry uses the `remainder`
      // bit-length even when remainder == 0. With remainder == 0
      // we end up appending an empty slice — match that.
      auto chunk = U32ToBitsLe(v[i], remainder);
      bits.insert(bits.end(), chunk.begin(), chunk.end());
    }
  }
  return BitsToBytesLe(bits);
}

}  // namespace primihub::pir::frodo
