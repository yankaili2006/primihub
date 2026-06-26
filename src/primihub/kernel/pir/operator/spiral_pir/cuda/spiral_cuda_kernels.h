/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * CUDA kernels for the SpiralPIR (USENIX'22) homomorphic hot paths
 * (openspec change primihub-pir-cuda-tiptoe, task 2.1): the GSW external
 * product and the Galois automorphism (query expansion), both in the NTT/CRT
 * domain that spiral-rs uses (crt_count residues, each a ~28-bit prime).
 *
 * Polynomials are stored as [crt_count][poly_len] uint64 (each coefficient <
 * its CRT modulus). The host wrappers manage device memory per call (correctness
 * first; keeping ciphertexts device-resident across the query-expansion tree,
 * and Montgomery reduction / a SIGMA-backed NTT, are perf follow-ups -- the
 * production path reuses pir-acc/SIGMA's CUDA-SEAL framework).
 *
 * Built only with a CUDA toolchain (nvcc + GPU); excluded from the default
 * CPU-only build (.50 has no GPU) via the disable_cuda gate (task 2.3).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_SPIRAL_CUDA_KERNELS_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_SPIRAL_CUDA_KERNELS_H_

#include <cstddef>
#include <cstdint>

namespace primihub::pir::spiral::cuda {

// Whether a usable CUDA device is present (cudaGetDeviceCount > 0).
bool CudaAvailable();

// GSW external product in the NTT/CRT domain. The GSW ciphertext is a
// 2 x rows_k matrix of polynomials; the input RLWE ct is gadget-decomposed into
// rows_k polynomials. For each output row r in {0,1}, CRT residue c, and
// coefficient i:
//   out[r][c][i] = sum_{k=0}^{rows_k-1} gsw[r][k][c][i] * decomp[k][c][i] mod q_c
//
// Layouts (flat, row-major): gsw [2][rows_k][crt_count][poly_len],
// decomp [rows_k][crt_count][poly_len], out [2][crt_count][poly_len],
// moduli [crt_count]. All pointers are host pointers. Inputs must already be
// reduced mod their residue's modulus.
void GswExternalProductNtt(std::uint64_t* out, const std::uint64_t* gsw,
                           const std::uint64_t* decomp,
                           const std::uint64_t* moduli, std::size_t rows_k,
                           std::size_t crt_count, std::size_t poly_len);

// Galois automorphism (x -> x^t) applied in the NTT domain as a slot
// permutation: out[c][i] = in[c][table[i]] for each residue c. `table` is the
// precomputed permutation for the chosen t (length poly_len). Host pointers.
void ApplyGaloisNtt(std::uint64_t* out, const std::uint64_t* in,
                    const std::size_t* table, std::size_t crt_count,
                    std::size_t poly_len);

}  // namespace primihub::pir::spiral::cuda

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_SPIRAL_PIR_CUDA_SPIRAL_CUDA_KERNELS_H_
