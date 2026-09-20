/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_modulus_switch — Spiral-free port of upstream
 * menonsamir/ypir@a73e550a src/modulus_switch.rs. Chunk 4 of the YPIR
 * port (see docs/pir/ypir-port-plan.md).
 *
 * Upstream defines `trait ModulusSwitch for PolyMatrixRaw` with two
 * methods over a 2x1 raw polynomial matrix (an RLWE ciphertext: two
 * rows of `poly_len` coefficients each):
 *   * switch(q_1, q_2) -> Vec<u8>           — rescale + bit-pack
 *   * recover(params, q_1, q_2, ct) -> Self — inverse
 *
 * This port is Spiral-free. The only upstream dependencies were
 * spiral_rs's `rescale` (pure integer arithmetic, ported here as
 * `Rescale`, mirroring menonsamir/spiral src/poly.cpp byte-for-byte)
 * and `write/read_arbitrary_bits` (already ported as WriteBits /
 * ReadBits in chunk 5a ypir_bits). The `PolyMatrixRaw` 2x1 container
 * is represented as two `poly_len`-length coefficient spans plus the
 * explicit `poly_len` / `modulus` params, so no Spiral C++ MatPoly /
 * Params facade is needed yet (those arrive when scheme.rs lands in
 * chunk 11).
 *
 * Bit-width quirk mirrored from upstream: row0 is packed with
 * q_1_bits = ceil(log2(q_2)) bits each, and row1 with
 * q_2_bits = ceil(log2(q_1)) — the cross-naming is upstream's, kept
 * for byte-exact fidelity.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MODULUS_SWITCH_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MODULUS_SWITCH_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Centred rounding division mapping a residue mod `inp_mod` to a
// residue mod `out_mod`. Standalone port of spiral_rs `rescale`
// (== menonsamir/spiral src/poly.cpp:rescale, nosign=false); uses
// __int128 internally to avoid overflow for moduli up to 2^63.
std::uint64_t Rescale(std::uint64_t a, std::uint64_t inp_mod,
                      std::uint64_t out_mod);

// Port of ModulusSwitch::switch for a 2x1 PolyMatrixRaw. `row0` /
// `row1` each point at `poly_len` coefficients (the two ciphertext
// rows). row0 coefficients are rescaled from `modulus` down to q_2,
// row1 down to q_1, then bit-packed. Returns the
// ceil((q_1_bits + q_2_bits) * poly_len / 8)-byte serialization.
std::vector<std::uint8_t> ModulusSwitchPack(
    const std::uint64_t* row0, const std::uint64_t* row1,
    std::size_t poly_len, std::uint64_t modulus,
    std::uint64_t q_1, std::uint64_t q_2);

// Inverse of ModulusSwitchPack (ModulusSwitch::recover). Unpacks
// `ciphertext` and rescales back up to `modulus`, writing `poly_len`
// coefficients into each of `*row0` / `*row1`. Returns false (leaving
// the outputs untouched) when `row0`/`row1` are null or when
// `ciphertext.size()` does not equal the expected
// ceil((q_1_bits + q_2_bits) * poly_len / 8).
bool ModulusSwitchRecover(
    const std::vector<std::uint8_t>& ciphertext,
    std::size_t poly_len, std::uint64_t modulus,
    std::uint64_t q_1, std::uint64_t q_2,
    std::vector<std::uint64_t>* row0,
    std::vector<std::uint64_t>* row1);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_MODULUS_SWITCH_H_
