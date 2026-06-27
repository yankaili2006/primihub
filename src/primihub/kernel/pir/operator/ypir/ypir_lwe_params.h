/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_lwe_params — C++ port of the LWEParams data structure from
 * upstream menonsamir/ypir@a73e550a src/lwe.rs. First sub-piece of
 * chunk 3 (lwe.rs); the LWEClient methods (encrypt / encrypt_many /
 * decrypt) need Spiral's DiscreteGaussian sampler + chunk 6's
 * convolution::negacyclic_matrix_u32, both of which are blocked on
 * the Spiral C++ port (task 3 / Phase 3 partial) and so are deferred
 * to chunk 3b.
 *
 * Naming — uses `LweParams` inside the `primihub::pir::ypir`
 * namespace to follow Google C++ style. This does NOT collide with
 * `primihub::pir::core::LweParams` (SimplePIR / DoublePIR variant);
 * the two structs have different shapes (YPIR's has `noise_width`
 * and `pt_modulus`; core's has `sigma` and `p`) and serve different
 * algorithms.
 *
 * Field-by-field correspondence with upstream Rust:
 *   pub n: usize           →  std::size_t n
 *   pub modulus: u64       →  std::uint64_t modulus
 *   pub pt_modulus: u64    →  std::uint64_t pt_modulus
 *   pub q2_bits: usize     →  std::size_t q2_bits
 *   pub noise_width: f64   →  double noise_width
 *
 * The Default() factory matches Rust's `impl Default for LWEParams`
 * exactly: n=1024, modulus=2^32, pt_modulus=256, q2_bits=28,
 * noise_width = 11 * sqrt(2*pi) ≈ 27.57291103. ScaleK() returns
 * modulus / pt_modulus (= 2^32 / 256 = 2^24 = 16777216 for default).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_PARAMS_H_

#include <cstddef>
#include <cstdint>

namespace primihub::pir::ypir {

// Default noise width — upstream Rust comment says
// `27.57291103, // 11 * sqrt(2*pi)`. Exposed as a constant for
// tests and downstream consumers that want to reason about
// noise budget independently of constructing a full LweParams.
constexpr double kDefaultLweNoiseWidth = 27.57291103;

struct LweParams {
  std::size_t n;
  std::uint64_t modulus;
  std::uint64_t pt_modulus;
  std::size_t q2_bits;
  double noise_width;

  // Factory mirroring Rust's `impl Default for LWEParams`.
  // Returns by value; the struct is trivially copyable.
  static LweParams Default();

  // scale_k from upstream — `self.modulus / self.pt_modulus`.
  // Integer division; for the default values this is
  // 2^32 / 256 = 16777216. Used by LWE encryption to encode
  // the plaintext into the high bits of the ciphertext domain.
  std::uint64_t ScaleK() const { return modulus / pt_modulus; }
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_LWE_PARAMS_H_
