/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_arith — P0 of the spiral_rs Params/PolyMatrix port (see
 * docs/pir/params-port-plan.md). Standalone C++ port of the modular
 * arithmetic / Barrett-reduction helpers from spiral-rs src/arith.rs
 * that Params::init and the PolyMatrix NTT-domain ops need.
 *
 * The Barrett constants (cr0, cr1) for a modulus q are the low / high
 * 64-bit limbs of floor(2^128 / q); Barrett reduction then computes
 * val mod q without a hardware divide. Ported faithfully and verified
 * against direct modulo and the upstream arith.rs test vectors.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_ARITH_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_ARITH_H_

#include <cstdint>
#include <utility>

namespace primihub::pir::ypir {

// Barrett constants for modulus q: {cr0, cr1} = the low and high 64-bit
// limbs of floor(2^128 / q). Mirrors spiral-rs get_barrett_crs.
std::pair<std::uint64_t, std::uint64_t> GetBarrettCrs(std::uint64_t modulus);

// Single-word Barrett reduction: returns input mod modulus, using only
// cr1 (= high limb of floor(2^128/modulus)). Mirrors barrett_raw_u64.
// Requires input < modulus^2 / 2^? — valid for input up to 2^64-1 with
// the standard one-subtraction correction.
std::uint64_t BarrettRawU64(std::uint64_t input, std::uint64_t cr1,
                            std::uint64_t modulus);

// 128-bit Barrett reduction: returns val mod modulus. Mirrors
// barrett_reduction_u128_raw (barrett_raw_u128 + one correction).
std::uint64_t BarrettReductionU128Raw(std::uint64_t modulus, std::uint64_t cr0,
                                      std::uint64_t cr1, __uint128_t val);

// a * b mod q for a CRT limb. crt_count==2 path: (a*b) fits u64 for the
// ~2^28 CRT moduli, reduced via Barrett; crt_count==1 path uses the full
// 128-bit product. Mirrors multiply_modular.
std::uint64_t MultiplyModular(std::uint64_t a, std::uint64_t b,
                              std::uint64_t modulus, std::uint64_t cr0,
                              std::uint64_t cr1, std::size_t crt_count);

// Floor log2 of a (the index of the highest set bit): Log2(1)=0, Log2(2)=1,
// Log2(255)=7, Log2(256)=8. Log2(0) returns 0. Mirrors spiral-rs log2, used
// to size the lazy-accumulation max_adds in ring hint generation.
std::uint64_t Log2(std::uint64_t a);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_ARITH_H_
