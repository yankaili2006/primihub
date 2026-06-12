/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_format — C++ port of the Spiral-free, base64-free subset
 * of upstream brave-experiments/frodo-pir@15573960 src/utils.rs
 * `pub mod format`. Bit-level pack/unpack helpers used by both
 * api.rs (Shard serialization) and db.rs (per-row LWE encoding).
 *
 * Position in the port plan (docs/pir/frodo-port-plan.md):
 *   chunk 1 — alongside frodo_lwe_consts. Both pure integer
 *   bit-manipulation, zero deps on Spiral or upstream Rust.
 *
 * Functions ported here:
 *   * U8ToBitsLe(byte) — 8 bools LSB-first.
 *   * U32ToBitsLe(x, bit_len) — `bit_len` bits of `x` LSB-first.
 *   * BitsToBytesLe(bits) — pack bits LSB-first into bytes.
 *   * BytesToBitsLe(bytes) — unpack bytes LSB-first into bits.
 *   * BitsToU32Le(bits, &out) — inverse of U32ToBitsLe; returns
 *     retcode::FAIL if `bits.size() / 8 > sizeof(uint32_t)`.
 *   * BytesFromU32Slice(v, entry_bit_len, total_bit_len) — pack
 *     a u32 slice via per-entry truncation to `entry_bit_len`,
 *     with the last entry truncated to `total_bit_len %
 *     entry_bit_len` bits.
 *
 * Functions intentionally NOT ported in chunk 1:
 *   * `u32_sized_bytes_from_vec` — Rust-specific TryInto; in C++
 *     we just check the size inside BitsToU32Le and copy.
 *   * `base64_from_u32_slice` — needs the `base64` crate; skipped.
 *
 * Bool representation: upstream returns `Vec<bool>`. C++ uses
 * `std::vector<uint8_t>` with 0/1 entries to avoid the
 * `std::vector<bool>` specialisation surprises (no `data()`,
 * proxy refs, etc.). Conversion is trivial at the boundary.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FORMAT_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FORMAT_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/primihub/common/common.h"

namespace primihub::pir::frodo {

// 8 bits of `byte` returned LSB-first. `out[i]` is 1 iff bit i of
// byte is set. Mirrors upstream `u8_to_bits_le(byte)`.
std::vector<std::uint8_t> U8ToBitsLe(std::uint8_t byte);

// First `bit_len` bits of u32 `x` in LSB-first order. If
// `bit_len > 32`, returns the full 32-bit expansion truncated to
// 32 bits (upstream Rust would panic on `bits[..bit_len]` when
// bit_len > 32; we keep a soft boundary).
std::vector<std::uint8_t> U32ToBitsLe(std::uint32_t x,
                                      std::size_t bit_len);

// Pack bits (each entry treated as 0 or non-zero -> 0 or 1) into
// bytes, LSB-first. Output size = ceil(bits.size() / 8).
// Mirrors upstream `bits_to_bytes_le`.
std::vector<std::uint8_t> BitsToBytesLe(
    const std::vector<std::uint8_t>& bits);

// Unpack bytes into bits, LSB-first. Output size = 8 * bytes.size().
// Mirrors upstream `bytes_to_bits_le`.
std::vector<std::uint8_t> BytesToBitsLe(
    const std::vector<std::uint8_t>& bytes);

// Inverse of U32ToBitsLe. `*out` receives the u32 reconstructed
// from `bits` (LSB-first). Returns retcode::FAIL with `err` set
// if `bits.size()` packs into more than 4 bytes (upstream Rust
// raises ErrorUnexpectedInputSize). On success `err` is left
// unchanged.
retcode BitsToU32Le(const std::vector<std::uint8_t>& bits,
                    std::uint32_t* out, std::string* err);

// Pack each u32 in `v` as `entry_bit_len` bits LSB-first, except
// the last entry which is packed as `total_bit_len % entry_bit_len`
// bits (matching upstream's per-entry truncation contract). The
// resulting bit-stream is then byte-packed via BitsToBytesLe.
// Returns an empty vector when `v` is empty.
std::vector<std::uint8_t> BytesFromU32Slice(
    const std::vector<std::uint32_t>& v, std::size_t entry_bit_len,
    std::size_t total_bit_len);

}  // namespace primihub::pir::frodo

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_FRODO_PIR_FRODO_FORMAT_H_
