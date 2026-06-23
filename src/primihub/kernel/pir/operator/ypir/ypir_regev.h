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
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
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

// Fresh Regev/RLWE sample: an encryption of zero under sk_reg. Returns a
// 2x1 NTT matrix p = [(-a).ntt(); (e + sk*a).ntt()] where a is uniform
// (rng_pub) and e is gaussian noise (rng). Mirrors client.rs
// get_reg_sample (dg passed in rather than re-Init per call -- identical).
// Decrypts as p_row1 + p_row0*sk == e (exact mod q).
PolyMatrixNTT GetRegSample(const NttContext& ctx, const DiscreteGaussian& dg,
                           const PolyMatrixRaw& sk_reg, ChaChaRng& rng,
                           ChaChaRng& rng_pub);

// Fresh Regev public key: a 2 x m matrix whose every column is an
// independent GetRegSample (encryption of zero). Mirrors client.rs
// get_fresh_reg_public_key.
PolyMatrixNTT GetFreshRegPublicKey(const NttContext& ctx,
                                   const DiscreteGaussian& dg,
                                   const PolyMatrixRaw& sk_reg, std::size_t m,
                                   ChaChaRng& rng, ChaChaRng& rng_pub);

// Expansion (query-unpacking) public params: num_exp key-switch keys,
// each a 2 x m_exp Regev encryption of automorph(sk, t_i) * gadget, where
// t_i = poly_len/2^i + 1. Mirrors client.rs raw_generate_expansion_params.
// Decrypts (col j) to e_j + (automorph(sk,t_i) * gadget)[j].
std::vector<PolyMatrixNTT> RawGenerateExpansionParams(
    const NttContext& ctx, const DiscreteGaussian& dg,
    const PolyMatrixRaw& sk_reg, std::size_t num_exp, std::size_t m_exp,
    ChaChaRng& rng, ChaChaRng& rng_pub);

// Regev-encrypt a 1x1 raw plaintext m (coeffs in [0, pt_modulus)) under
// sk_reg: a 2x1 ciphertext = GetRegSample + Delta*m in row 1, where
// Delta*m[z] = Rescale(m[z], pt_modulus, modulus). Test-and-operator
// helper (upstream encrypts via the spiral Client; this is the YPIR-local
// equivalent). Decrypts via RegevDecrypt.
PolyMatrixNTT RegevEncrypt(const NttContext& ctx, const DiscreteGaussian& dg,
                           const PolyMatrixRaw& sk_reg, const PolyMatrixRaw& m,
                           ChaChaRng& rng, ChaChaRng& rng_pub);

// Decrypt a 2x1 Regev ciphertext under sk_reg: phase = ct_row0*sk + ct_row1,
// then per coeff Rescale(phase, modulus, pt_modulus). Returns 1x1 raw.
PolyMatrixRaw RegevDecrypt(const NttContext& ctx, const PolyMatrixRaw& sk_reg,
                           const PolyMatrixNTT& ct);

// Homomorphic Galois automorphism X->X^t on a 2x1 Regev ciphertext via a
// gadget key-switch under pub_param (= RawGenerateExpansionParams[level]
// for the same t). Mirrors packing.rs homomorphic_automorph. t_exp must
// equal pub_param.cols (the gadget dimension). Decrypts to the plaintext
// automorph: dec(HomomorphicAutomorph(enc(m), t)) == decode(automorph(encode(m))).
PolyMatrixNTT HomomorphicAutomorph(const NttContext& ctx, std::size_t t,
                                   std::size_t t_exp, const PolyMatrixNTT& ct,
                                   const PolyMatrixNTT& pub_param);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_REGEV_H_
