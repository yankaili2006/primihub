/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_scheme — chunk 11 of the YPIR port: the scheme/param seed machinery
 * that the server (10c+) and client query generation depend on. Ports:
 *   - scheme.rs STATIC_PUBLIC_SEED / STATIC_SEED_2 + client.rs get_seed
 *   - params.rs YPIRParams + the GetQPrime trait (get_q_prime_1/2) on Params,
 *     including spiral-rs Q2_VALUES.
 *
 * (The SEED_0/SEED_1 public-seed indices are kSeed0/kSeed1 in ypir_server.h,
 * defined there when multiply_with_db_ring first needed them.)
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SCHEME_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SCHEME_H_

#include <array>
#include <cstdint>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"

namespace primihub::pir::ypir {

// scheme.rs STATIC_PUBLIC_SEED = [0u8; 32].
std::array<std::uint8_t, 32> StaticPublicSeed();

// scheme.rs STATIC_SEED_2 = [2, 0, 0, ..., 0] (first byte 2, rest 0).
std::array<std::uint8_t, 32> StaticSeed2();

// client.rs get_seed: STATIC_PUBLIC_SEED with byte 0 overwritten by
// public_seed_idx (so get_seed(SEED_0) is all-zero, get_seed(SEED_1) is
// [1, 0, ..., 0]).
std::array<std::uint8_t, 32> GetSeed(std::uint8_t public_seed_idx);

// params.rs YPIRParams.
struct YPIRParams {
  bool is_simplepir = false;
};

// params.rs GetQPrime for Params. q_prime_1 is the fixed 1<<20 second-row
// modulus; q_prime_2 is the first-row modulus: the full product modulus when
// q2_bits == modulus_log2, else the spiral-rs Q2_VALUES[q2_bits] entry.
std::uint64_t GetQPrime1(const Params& params);
std::uint64_t GetQPrime2(const Params& params);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_SCHEME_H_
