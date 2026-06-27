/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_params — P1 of the spiral_rs Params port (see
 * docs/pir/params-port-plan.md). A standalone C++ port of spiral-rs
 * src/params.rs `Params` as a pure runtime scalar config struct +
 * `Init`. Deliberately holds NO NTT engine: spiral_rs's `ntt_tables`
 * are replaced by HEXL NTT contexts built separately in P2 (PolyMatrix),
 * which keeps Params dependency-free (only the P0 barrett helpers) and
 * fully testable without HEXL. `scratch` is dropped (an internal buffer).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PARAMS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PARAMS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

inline constexpr std::size_t kMaxModuli = 4;  // spiral_rs MAX_MODULI

struct Params {
  std::size_t poly_len = 0;
  std::size_t poly_len_log2 = 0;
  std::size_t crt_count = 0;
  std::array<std::uint64_t, kMaxModuli> moduli{};
  std::uint64_t modulus = 1;       // product of moduli
  std::uint64_t modulus_log2 = 0;  // ceil(log2(modulus))

  // Barrett constants (low/high limbs of floor(2^128/q)) per CRT modulus
  // and for the product modulus.
  std::array<std::uint64_t, kMaxModuli> barrett_cr_0{};
  std::array<std::uint64_t, kMaxModuli> barrett_cr_1{};
  std::uint64_t barrett_cr_0_modulus = 0;
  std::uint64_t barrett_cr_1_modulus = 0;

  // CRT idempotents (crt_count==2): mod0_inv_mod1 == moduli[0] *
  // (moduli[0]^{-1} mod moduli[1]); analogously mod1_inv_mod0.
  std::uint64_t mod0_inv_mod1 = 0;
  std::uint64_t mod1_inv_mod0 = 0;

  double noise_width = 0.0;
  std::size_t n = 0;
  std::uint64_t pt_modulus = 0;
  std::uint64_t q2_bits = 0;

  std::size_t t_conv = 0;
  std::size_t t_exp_left = 0;
  std::size_t t_exp_right = 0;
  std::size_t t_gsw = 0;

  bool expand_queries = false;
  std::size_t db_dim_1 = 0;
  std::size_t db_dim_2 = 0;
  std::size_t instances = 0;
  std::size_t db_item_size = 0;
  std::size_t version = 0;

  // Mirrors spiral-rs Params::init (minus ntt_tables/scratch). `moduli`
  // has crt_count (1 or 2) entries.
  static Params Init(std::size_t poly_len, const std::vector<std::uint64_t>& moduli,
                     double noise_width, std::size_t n, std::uint64_t pt_modulus,
                     std::uint64_t q2_bits, std::size_t t_conv,
                     std::size_t t_exp_left, std::size_t t_exp_right,
                     std::size_t t_gsw, bool expand_queries, std::size_t db_dim_1,
                     std::size_t db_dim_2, std::size_t instances,
                     std::size_t db_item_size, std::size_t version);
};

inline constexpr std::uint64_t kMinQ2Bits = 14;  // spiral_rs MIN_Q2_BITS

// P5: the expansion-mode YPIR parameter preset, mirroring ypir
// internal_params_for. Fixes n=1, t_gsw=3, t_conv=4, t_exp_right=2,
// instances=1, poly_len=2048, noise_width=6.4, expand_queries=true,
// version=0; q2_bits is clamped to >= kMinQ2Bits; db_item_size =
// poly_len * ceil(log2(p)) / 8. db_dim_1/2 come from nu_1/nu_2.
Params ParamsForExpansion(std::size_t nu_1, std::size_t nu_2, std::uint64_t p,
                          std::uint64_t q2_bits, std::size_t t_exp_left,
                          const std::vector<std::uint64_t>& moduli);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_PARAMS_H_
