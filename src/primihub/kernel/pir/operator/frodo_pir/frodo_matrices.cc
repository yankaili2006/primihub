/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

#include <cstdint>
#include <sstream>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_prng.h"

namespace primihub::pir::frodo {

namespace {

#if defined(__x86_64__) || defined(_M_X64)
// AVX2 inner kernel: 8-wide u32 wrapping mul + add reduction.
// __attribute__((target("avx2"))) lets this compile under the
// default linux_x86_64 toolchain (.bazelrc only enables SSE4.1).
// Runtime dispatch via __builtin_cpu_supports keeps non-AVX2
// hardware on the scalar path.
__attribute__((target("avx2")))
std::uint32_t VecMultU32U32Avx2(const std::uint32_t* a,
                                const std::uint32_t* b,
                                std::size_t n) {
  __m256i acc = _mm256_setzero_si256();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256i va = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(b + i));
    __m256i prod = _mm256_mullo_epi32(va, vb);  // wrap mod 2^32
    acc = _mm256_add_epi32(acc, prod);           // wrap mod 2^32
  }
  // Horizontal sum of 8 u32 lanes; explicit wrapping carries.
  alignas(32) std::uint32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
  std::uint32_t hsum = 0u;
  for (int j = 0; j < 8; ++j) hsum += lanes[j];
  // Scalar tail.
  for (; i < n; ++i) hsum += a[i] * b[i];
  return hsum;
}
#endif

std::uint32_t VecMultU32U32Inner(const std::uint32_t* a,
                                 const std::uint32_t* b,
                                 std::size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
  // Use AVX2 path for any length that crosses one 8-lane block
  // (smaller inputs hit cache effects + dispatch overhead).
  if (n >= 16 && __builtin_cpu_supports("avx2")) {
    return VecMultU32U32Avx2(a, b, n);
  }
#endif
  std::uint32_t acc = 0u;
  for (std::size_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

}  // namespace

std::vector<std::vector<std::uint32_t>> SwapMatrixFmt(
    const std::vector<std::vector<std::uint32_t>>& matrix) {
  if (matrix.empty()) {
    return {};
  }
  const std::size_t height = matrix.size();
  const std::size_t width = matrix[0].size();
  std::vector<std::vector<std::uint32_t>> swapped(width);
  for (auto& col : swapped) {
    col.reserve(height);
  }
  for (const auto& row : matrix) {
    // Upstream contract: every row has the same width as matrix[0].
    // We mirror that — no defensive resize beyond width.
    for (std::size_t i = 0; i < width; ++i) {
      swapped[i].push_back(row[i]);
    }
  }
  return swapped;
}

std::vector<std::uint32_t> GetMatrixSecondAt(
    const std::vector<std::vector<std::uint32_t>>& matrix,
    std::size_t secidx) {
  if (matrix.empty() || secidx >= matrix[0].size()) {
    return {};
  }
  std::vector<std::uint32_t> col;
  col.reserve(matrix.size());
  for (const auto& row : matrix) {
    col.push_back(row[secidx]);
  }
  return col;
}

retcode VecMultU32U32(const std::vector<std::uint32_t>& row,
                      const std::vector<std::uint32_t>& col,
                      std::uint32_t* out, std::string* err) {
  if (out == nullptr) {
    if (err) *err = "VecMultU32U32: out must be non-null";
    return retcode::FAIL;
  }
  if (row.size() != col.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "VecMultU32U32: row_len: " << row.size()
          << ", col_len: " << col.size()
          << ". Upstream raises ErrorUnexpectedInputSize here.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  const std::size_t n = row.size();
  const std::uint32_t* a = row.data();
  const std::uint32_t* b = col.data();
  std::uint32_t acc = VecMultU32U32Inner(a, b, n);
  *out = acc;
  return retcode::SUCCESS;
}



std::vector<std::vector<std::uint32_t>> GenerateLweMatrixFromSeed(
    const SeedBytes& seed, std::size_t lwe_dim, std::size_t width) {
  // Upstream:
  //   let mut a = Vec::with_capacity(width);
  //   let mut rng = get_seeded_rng(seed);
  //   for _ in 0..width {
  //     let mut v = Vec::with_capacity(lwe_dim);
  //     for _ in 0..lwe_dim { v.push(rng.next_u32()); }
  //     a.push(v);
  //   }
  //   a
  SeededRng rng(seed);
  std::vector<std::vector<std::uint32_t>> a;
  a.reserve(width);
  // Bulk-fill each row in one EVP_EncryptUpdate call rather
  // than `lwe_dim` separate NextU32 invocations. On x86_64
  // little-endian (the only build target — .bazelrc gates SSE)
  // reading the keystream bytes into a u32 buffer directly
  // matches the LE assembly done by NextU32. Cross-checked by
  // FrodoMatricesTest GenerateLweMatrixFromSeed_ColumnOrder
  // MatchesUpstream which pins this against direct SeededRng.
  static_assert(sizeof(std::uint32_t) == 4,
                "frodo_matrices: u32 must be 4 bytes");
  for (std::size_t col = 0; col < width; ++col) {
    std::vector<std::uint32_t> v(lwe_dim, 0u);
    rng.FillBytesBulk(reinterpret_cast<std::uint8_t*>(v.data()),
                       lwe_dim * sizeof(std::uint32_t));
    a.push_back(std::move(v));
  }
  return a;
}



ColMajorMatrix GenerateLweMatrixFromSeedFlat(
    const SeedBytes& seed, std::size_t lwe_dim, std::size_t width) {
  // Empty-shape shortcut. Mirrors the per-column overload's
  // "lwe_dim==0 || width==0 -> empty" boundary so the migration
  // is drop-in.
  if (lwe_dim == 0 || width == 0) {
    return ColMajorMatrix{};
  }
  // NoInit: SeededRng is about to overwrite every byte via the
  // ChaCha20 keystream. The zero-init from the regular ctor
  // would be wasted writes.
  ColMajorMatrix m(/*height=*/lwe_dim, /*width=*/width,
                   ColMajorMatrix::NoInit{});
  static_assert(sizeof(std::uint32_t) == 4,
                "frodo_matrices: u32 must be 4 bytes");
  // ColMajorMatrix's NoInit ctor goes through vector::resize which
  // value-initialises uint32_t to 0 on libstdc++ — so the storage
  // is already a zero buffer. Use FillKeystreamBulk which skips
  // the internal memset that FillBytesBulk would otherwise repeat.
  // For width=m and lwe_dim=512 the buffer is ~2 GB at N=1M;
  // a redundant memset would cost ~200 ms of write bandwidth.
  // OpenSSL's ChaCha20 AVX2 path saturates at ~3 GB/s on
  // Broadwell so the EVP step itself is bandwidth-bound, not
  // CPU-bound.
  SeededRng rng(seed);
  rng.FillKeystreamBulk(
      reinterpret_cast<std::uint8_t*>(m.raw_data()),
      m.total_u32s() * sizeof(std::uint32_t));
  // Byte-for-byte equivalence with the per-column overload comes
  // from the layout invariant: column c lives at
  // storage[c * lwe_dim .. (c+1) * lwe_dim), which is exactly
  // where the per-column form would have placed it after
  // append. Pinned by
  // FrodoMatricesTest::GenerateLweMatrixFromSeedFlat_MatchesPerColumn
  // _Width2049.
  return m;
}



namespace {

// Upstream:
//   const TERNARY_INTERVAL_SIZE: u32 = (u32::MAX - 2) / 3;
//   const TERNARY_REJECTION_SAMPLING_MAX: u32 = TERNARY_INTERVAL_SIZE * 3;
// We mirror these byte-for-byte.
constexpr std::uint32_t kTernaryIntervalSize =
    (0xFFFFFFFFu - 2u) / 3u;
constexpr std::uint32_t kTernaryRejectionMax =
    kTernaryIntervalSize * 3u;

}  // namespace

std::uint32_t RandomTernary() {
  std::uint32_t val = os_rng::NextU32();
  while (val > kTernaryRejectionMax) {
    val = os_rng::NextU32();
  }
  // val now in [0, 3*kTernaryIntervalSize]; trichotomy mirrors
  // upstream (note inclusive lower-bound on the first interval).
  if (val > kTernaryIntervalSize &&
      val <= kTernaryIntervalSize * 2u) {
    return 1u;
  }
  if (val > kTernaryIntervalSize * 2u) {
    return 0xFFFFFFFFu;  // upstream uses u32::MAX (= -1 mod 2^32)
  }
  return 0u;
}

std::vector<std::uint32_t> RandomTernaryVector(std::size_t width) {
  std::vector<std::uint32_t> out;
  out.reserve(width);
  for (std::size_t i = 0; i < width; ++i) {
    out.push_back(RandomTernary());
  }
  return out;
}

}  // namespace primihub::pir::frodo
