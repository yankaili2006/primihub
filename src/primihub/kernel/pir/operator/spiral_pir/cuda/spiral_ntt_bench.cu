// Warm kernel-only microbench for the SpiralPIR negacyclic NTT: radix-2 (the
// reference DIT) vs radix-4 (fused), batched device-resident, at spiral's real
// params (poly_len=2048, crt=2, the two ~28-bit DEFAULT_MODULI). Reports ms per
// forward+inverse over the whole batch + NTTs/s (one "NTT" = one residue
// transform), excluding H2D/D2H. Sweeps batch size to show GPU fill.
#include <cstdint>
#include <cstdio>

namespace primihub::pir::spiral::cuda {
double BenchNttKernelMs(std::size_t num_polys, const std::uint64_t* moduli,
                        std::size_t crt, std::size_t N, int iters, bool radix4);
double BenchEndToEndMs(std::size_t num_polys, const std::uint64_t* moduli,
                       std::size_t crt, std::size_t N, int iters, int mode);
}

int main(int argc, char** argv) {
  using primihub::pir::spiral::cuda::BenchNttKernelMs;
  const std::uint64_t moduli[2] = {268369921ull, 249561089ull};
  const std::size_t crt = 2, N = 2048;
  const int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
  const std::size_t batches[] = {1, 64, 256, 1024, 4096};

  std::printf("SpiralPIR NTT microbench  (N=%zu, crt=%zu, iters=%d, fwd+inv per iter)\n",
              N, crt, iters);
  std::printf("%8s | %12s | %12s | %7s | %14s\n", "polys", "r2 ms/iter",
              "r4 ms/iter", "speedup", "r4 NTTs/s");
  std::printf("---------|--------------|--------------|---------|---------------\n");
  for (std::size_t b : batches) {
    const double r2 = BenchNttKernelMs(b, moduli, crt, N, iters, /*radix4=*/false) / iters;
    const double r4 = BenchNttKernelMs(b, moduli, crt, N, iters, /*radix4=*/true) / iters;
    // forward+inverse over b polys x crt residues = 2*b*crt residue-NTTs per iter.
    const double ntts = 2.0 * b * crt;
    const double rate = ntts / (r4 / 1e3);
    std::printf("%8zu | %12.4f | %12.4f | %6.2fx | %14.3e\n", b, r2, r4,
                r2 / r4, rate);
  }

  using primihub::pir::spiral::cuda::BenchEndToEndMs;
  const int e2e_iters = 20;
  std::printf(
      "\nEnd-to-end (incl. H2D/D2H + alloc), fwd+inv, %d iters:\n", e2e_iters);
  std::printf("%8s | %16s | %16s | %8s\n", "polys", "original ms",
              "batched ms", "speedup");
  std::printf("---------|------------------|------------------|---------\n");
  for (std::size_t b : {64ul, 256ul, 1024ul}) {
    const double old_ms = BenchEndToEndMs(b, moduli, crt, N, e2e_iters, 0);
    const double new_ms = BenchEndToEndMs(b, moduli, crt, N, e2e_iters, 1);
    std::printf("%8zu | %16.3f | %16.3f | %6.1fx\n", b, old_ms, new_ms,
                old_ms / new_ms);
  }
  return 0;
}
