/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_regev -- the Regev/RLWE public-key layer ported from upstream
 * src/client.rs (chunk 12). Starts with the two randomized PolyMatrixRaw
 * constructors (random_rng / noise) that the public-key sampling
 * (get_reg_sample / get_fresh_reg_public_key / raw_generate_expansion_params)
 * is built on. YPIR has no cross-impl wire boundary, so these need only be
 * internally self-consistent (encrypt/decrypt inverses), not byte-identical
 * to the Rust reference -- but the RNG consumption order IS mirrored exactly.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_REGEV_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_REGEV_H_

#include <cstddef>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Uniform raw poly matrix: one rng.NextU64() per coefficient, reduced
// mod p.modulus. Mirrors spiral-rs PolyMatrixRaw::random_rng (sample_iter
// over Standard u64). Consumes rows*cols*poly_len u64 from rng.
PolyMatrixRaw RandomRngRaw(const Params& p, std::size_t rows, std::size_t cols,
                           ChaChaRng& rng);

// Discrete-Gaussian noise raw poly matrix: each coefficient is
// dg.Sample(p.modulus, rng.NextU64()). Mirrors spiral-rs
// PolyMatrixRaw::noise -> DiscreteGaussian::sample_matrix.
PolyMatrixRaw NoiseRaw(const Params& p, std::size_t rows, std::size_t cols,
                       const DiscreteGaussian& dg, ChaChaRng& rng);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_REGEV_H_
