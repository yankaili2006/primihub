// Standalone GPU validation of the SpiralPIR CUDA kernels (task 2.1): GSW
// external product + Galois automorphism vs a CPU reference, at spiral's real
// params (poly_len=2048, crt_count=2, the two ~28-bit DEFAULT_MODULI).
#include "spiral_cuda_kernels.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace primihub::pir::spiral::cuda;

int main() {
  const std::uint64_t moduli[2] = {268369921ull, 249561089ull};
  const std::size_t crt = 2, pl = 2048, rows_k = 8;  // gadget dim 2*t=8

  // Deterministic pseudo-random inputs, each reduced mod its residue's modulus.
  auto rnd = [](std::uint64_t x) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x;
  };
  std::vector<std::uint64_t> gsw(2 * rows_k * crt * pl), dec(rows_k * crt * pl);
  std::uint64_t s = 0x1234567;
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t k = 0; k < rows_k; ++k)
      for (std::size_t c = 0; c < crt; ++c)
        for (std::size_t i = 0; i < pl; ++i) {
          s = rnd(s + 1);
          gsw[((r * rows_k + k) * crt + c) * pl + i] = s % moduli[c];
        }
  for (std::size_t k = 0; k < rows_k; ++k)
    for (std::size_t c = 0; c < crt; ++c)
      for (std::size_t i = 0; i < pl; ++i) {
        s = rnd(s + 1);
        dec[(k * crt + c) * pl + i] = s % moduli[c];
      }

  // --- GSW external product ---
  std::vector<std::uint64_t> got(2 * crt * pl, 0), ref(2 * crt * pl, 0);
  GswExternalProductNtt(got.data(), gsw.data(), dec.data(), moduli, rows_k, crt, pl);
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t c = 0; c < crt; ++c)
      for (std::size_t i = 0; i < pl; ++i) {
        unsigned __int128 acc = 0;
        for (std::size_t k = 0; k < rows_k; ++k) {
          const std::uint64_t g = gsw[((r * rows_k + k) * crt + c) * pl + i];
          const std::uint64_t d = dec[(k * crt + c) * pl + i];
          acc += static_cast<unsigned __int128>(g) * d;
        }
        ref[(r * crt + c) * pl + i] = static_cast<std::uint64_t>(acc % moduli[c]);
      }
  std::size_t mismatch = 0;
  for (std::size_t i = 0; i < got.size(); ++i)
    if (got[i] != ref[i]) ++mismatch;
  std::printf("GSW external product (2x%zu, poly_len=%zu, crt=%zu): %zu mismatches -> %s\n",
              rows_k, pl, crt, mismatch, mismatch == 0 ? "PASS" : "FAIL");

  // --- Galois automorphism (NTT-slot permutation) ---
  std::vector<std::size_t> table(pl);
  for (std::size_t i = 0; i < pl; ++i) table[i] = (i * 5 + 3) % pl;  // a permutation
  std::vector<std::uint64_t> in(crt * pl), gout(crt * pl, 0), gref(crt * pl, 0);
  for (std::size_t c = 0; c < crt; ++c)
    for (std::size_t i = 0; i < pl; ++i) { s = rnd(s + 1); in[c * pl + i] = s % moduli[c]; }
  ApplyGaloisNtt(gout.data(), in.data(), table.data(), crt, pl);
  for (std::size_t c = 0; c < crt; ++c)
    for (std::size_t i = 0; i < pl; ++i) gref[c * pl + i] = in[c * pl + table[i]];
  std::size_t gmis = 0;
  for (std::size_t i = 0; i < gout.size(); ++i)
    if (gout[i] != gref[i]) ++gmis;
  std::printf("Galois automorphism (poly_len=%zu, crt=%zu): %zu mismatches -> %s\n",
              pl, crt, gmis, gmis == 0 ? "PASS" : "FAIL");

  return (mismatch == 0 && gmis == 0) ? 0 : 1;
}
