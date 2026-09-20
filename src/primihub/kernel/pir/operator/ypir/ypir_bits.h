/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_bits — Spiral-free bit-twiddling helpers from upstream
 * menonsamir/ypir@a73e550a src/bits.rs. Chunk 5a of the YPIR port
 * (see docs/pir/ypir-port-plan.md): the four pure bit-manipulation
 * functions can port today; the two AlignedMemory64 byte-reinterpret
 * helpers (`as_bytes`, `as_bytes_mut`) are deferred to chunk 5b
 * because they require the Spiral C++ aligned-memory facade
 * (task 3 / Phase 3 partial).
 *
 * What's here:
 *   * WriteBits(data, val, bit_offs, num_bits) — write `num_bits`
 *     LSBs of `val` into `data` starting at bit offset `bit_offs`.
 *     OR-merges into existing bits (caller responsible for zeroing
 *     the destination if a clean overwrite is wanted, mirroring
 *     upstream).
 *   * ReadBits(data, bit_offs, num_bits) — read up to 64 contiguous
 *     bits from `data` and return as u64. Returns 0 when
 *     `num_bits == 0` or `num_bits > 64` (upstream Rust panics; we
 *     prefer a quiet zero at our boundary).
 *   * U64sToContiguousBytes(data, inp_mod_bits) — packed serialize
 *     a slice of u64s where each value uses `inp_mod_bits` bits.
 *     Output size = ceil(data.len() * inp_mod_bits / 8).
 *   * ContiguousBytesToU64s(data, out_mod_bits) — inverse of
 *     U64sToContiguousBytes; reads `data.len() * 8 / out_mod_bits`
 *     values of `out_mod_bits` bits each.
 *
 * Roundtrip invariant tested in ypir_bits_test:
 *   ContiguousBytesToU64s(U64sToContiguousBytes(v, k), k) == v
 *   for every v whose entries are < 2^k.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_BITS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_BITS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Write `num_bits` LSBs of `val` into `data` starting at bit offset
// `bit_offs` (counting from the LSB of byte 0). Mirrors upstream's
// OR-into semantics — callers wanting a clean overwrite must zero
// the destination range first.
//
// Out-of-range writes (running past `data.size()`) silently stop;
// upstream's `while ... && byte_index < data.len()` produces the
// same behaviour.
void WriteBits(std::uint8_t* data, std::size_t data_len,
               std::uint64_t val, std::size_t bit_offs,
               std::size_t num_bits);

// Read up to 64 contiguous bits from `data` starting at `bit_offs`
// and return as a u64. Returns 0 when num_bits is 0 or > 64 (the
// upstream Rust panics; we keep a soft boundary).
std::uint64_t ReadBits(const std::uint8_t* data, std::size_t data_len,
                       std::size_t bit_offs, std::size_t num_bits);

// Packed serialize: each `data[i]` written as `inp_mod_bits` bits,
// in order. Output size is `ceil(data.size() * inp_mod_bits / 8)`.
// Callers must ensure each `data[i] < (1 << inp_mod_bits)` for the
// roundtrip to recover them — WriteBits only inspects the low
// `inp_mod_bits` bits via the mask, so excess high bits are silently
// dropped.
std::vector<std::uint8_t> U64sToContiguousBytes(
    const std::vector<std::uint64_t>& data, std::size_t inp_mod_bits);

// Inverse of U64sToContiguousBytes. Returns
// `data.size() * 8 / out_mod_bits` values (integer-divide truncation
// matches upstream `(data.len() * 8) / out_mod_bits`). Returns an
// empty vector when out_mod_bits is 0 or > 64.
std::vector<std::uint64_t> ContiguousBytesToU64s(
    const std::vector<std::uint8_t>& data, std::size_t out_mod_bits);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_BITS_H_
