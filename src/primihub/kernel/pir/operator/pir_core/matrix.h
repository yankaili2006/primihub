/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::core::Matrix — uint32 row-major dense matrix shared by
 * the SimplePIR / DoublePIR / YPIR C++ ports. Bridges shape-aware C++
 * API to the @simplepir C matrix-multiplication kernels (pir/pir.c)
 * established by tasks 5.1+5.2 and activated by 5.5 / 7.2 / 7.3
 * scaffolds.
 *
 * Layout — ROW-MAJOR uint32_t, matching upstream simplepir's matrix.go
 * (Elem = C.uint32_t, Data[i*Cols+j]). Picking the same layout means
 * the future Go-to-C++ port can use the matrix abstraction without
 * re-deriving indexing semantics.
 *
 * Vendoring — like the per-operator runtime facades (commit dc037df7),
 * Matrix has two compile modes selected by PIR_PIR_CORE_REAL set by
 * the BUILD select() under --define=enable_pir_core_real=1:
 *
 *   * Defined: Mul / MulVec / Transpose forward into upstream C
 *     kernels and produce real results.
 *   * Undefined: those methods return retcode::FAIL with a populated
 *     `err` mentioning the activation flag. Pure-arithmetic methods
 *     (Get/Set/Add/Sub/Reduce) always work — they are header-only
 *     and do not touch the kernel.
 *
 * Scope at this revision — the FIRST shared-infra port chunk. Covers
 * the methods SimplePIR / DoublePIR will use during their first port
 * pass: shape, factories (Zero / Rand / FromBytes), arithmetic (Add /
 * Sub / Reduce / DropLastRows), and the three kernel bridges
 * (Mul / MulVec / Transpose). Skipped intentionally for later: packed
 * variants (Squish / Unsquish / Expand / Contract — needed for DoublePIR
 * compression), Gaussian sampling (needs the LWE noise distribution),
 * row / column selection helpers. Add them when the operator port
 * actually calls them so we keep the abstraction lean.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_MATRIX_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_MATRIX_H_

#include <cstdint>
#include <string>
#include <vector>

#include "src/primihub/common/common.h"

namespace primihub::pir::core {

// True iff this build links against @simplepir//:simplepir_c_kernels.
// When false, kernel-calling methods (Mul / MulVec / Transpose) return
// retcode::FAIL.
extern const bool kPirCoreKernelsVendored;

class Matrix {
 public:
  Matrix() = default;
  Matrix(uint64_t rows, uint64_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols, 0) {}

  uint64_t rows() const { return rows_; }
  uint64_t cols() const { return cols_; }
  uint64_t size() const { return rows_ * cols_; }
  bool empty() const { return data_.empty(); }

  // Direct accessors. Out-of-bounds is a programmer error; we trade the
  // upstream panic for an LOG(FATAL) so caller stacks land in glog.
  uint32_t Get(uint64_t i, uint64_t j) const;
  void Set(uint64_t i, uint64_t j, uint32_t value);

  // Raw data window for kernel bridges. Pointer ownership stays with
  // this Matrix; callers must not free or resize via the pointer.
  uint32_t* mutable_data() { return data_.data(); }
  const uint32_t* data() const { return data_.data(); }

  // Factory helpers. Zeros() lives in the ctor; the others return new
  // Matrix instances by value (cheap thanks to std::vector move).
  static Matrix Zeros(uint64_t rows, uint64_t cols) {
    return Matrix(rows, cols);
  }
  // Uniform random in [0, 2^logmod). Uses std::mt19937_64 seeded from
  // std::random_device — fine for unit tests, NOT cryptographically
  // strong. Caller code that needs cryptographic randomness (LWE
  // secrets, queries) must inject its own RNG.
  static Matrix UniformRandom(uint64_t rows, uint64_t cols, uint32_t logmod);

  // In-place scalar / matrix arithmetic. Dimension mismatches in
  // MatrixAdd / MatrixSub LOG(FATAL); callers should pre-check.
  void MatrixAdd(const Matrix& other);
  void MatrixSub(const Matrix& other);
  void ScalarAdd(uint32_t value);
  void ScalarSub(uint32_t value);
  void ReduceMod(uint32_t modulus);

  // Drops the last `n` rows. Used by simplepir's MulVecPacked padding
  // workaround (it allocates Rows+8 then drops 8 back). Cheap — only
  // shrinks the underlying vector + updates rows_.
  void DropLastRows(uint64_t n);

  // AppendZeros — appends `n` rows of zeros to a COLUMN VECTOR (cols
  // must equal 1). Mirrors upstream simplepir's matrix.go AppendZeros,
  // which internally calls Concat(MatrixZeros(n, 1)). Used by Query
  // paths that pad the M x 1 ciphertext up to a multiple of
  // info.squishing so MulVecPacked sees a length divisible by 3.
  // LOG(FATAL) when cols_ != 1.
  void AppendZeros(uint64_t n);

  // ConcatCols — reshapes columns by folding into rows. Pre: cols_ %
  // n == 0. Post: rows_ becomes rows_ * n, cols_ becomes cols_ / n.
  // The j-th input column is placed at output column (j / n), at row
  // offset (j % n) * rows_ (i.e. column groups of size n are stacked
  // vertically). Mirrors upstream simplepir's matrix.go ConcatCols(n)
  // — used by DoublePIR Setup to pack the per-database H1 hint
  // matrix before the second-level multiply. Pure arithmetic — works
  // in both stub and vendored modes. n == 1 is a no-op; n == 0 LOG-
  // FATALs (would divide by zero).
  void ConcatCols(uint64_t n);

  // Expand(p, delta) — base-p digit decomposition into delta digits
  // per element, growing rows by a factor of delta with cols
  // unchanged. Input m x k -> output (m * delta) x k with the
  // permutation
  //     new[i * delta + f, j] = (old[i, j] / p^f) % p  -  p / 2
  // The trailing "-p/2" matches upstream simplepir's centered
  // representation (relies on uint32 underflow wrap-around — the same
  // pattern Database::ScalarSub uses for the centered-DB shift).
  //
  // Used by DoublePIR Setup to expand each H1 cell into its base-p
  // digits before the second-level multiply against A2, and by the
  // TransposeAndExpandAndConcatColsAndSquish fused upstream helper
  // (which we will assemble out of these primitives rather than
  // porting the C kernel directly).
  //
  // p == 0 LOG-FATALs (would divide by zero). p == 1 LOG-FATALs
  // (every digit would be 0). delta == 0 LOG-FATALs (would produce a
  // zero-row matrix and lose all data). Pure arithmetic — works in
  // both stub and vendored modes.
  void Expand(uint64_t p, uint64_t delta);

  // Contract(p, delta) — base-p digit re-aggregation, the upstream
  // simplepir matrix.go Contract verbatim. Pre: rows % delta == 0
  // (LOG-FATAL otherwise). Input (n * delta) x k -> output n x k with
  //     new[i, j] = Sum_{f=0..delta-1} p^f * ((old[i*delta+f, j] + p/2) % p)
  // The (+p/2) % p re-centers the digit before reconstruction.
  //
  // NOT a clean inverse of Expand for arbitrary p: the uint32
  // underflow Expand uses to encode negative digits gets re-read as
  // a large uint64 here, introducing a per-digit offset of
  // (2^32 mod p) in the recentered value. Round-trip is exact only
  // when p divides 2^32 (i.e., p is a power of 2). DoublePIR uses
  // arbitrary p (e.g. 929, 781) and handles this by applying
  // Contract within an LWE protocol context where the offset cancels
  // — NOT as a plaintext decoder. Tests assert the upstream
  // algorithm verbatim rather than round-trip identity.
  //
  // Pure arithmetic — works in both stub and vendored modes.
  void Contract(uint64_t p, uint64_t delta);

  // Concat — row-append. Mirrors upstream simplepir matrix.go Concat
  // (which appends `other.Data` after `m.Data` when cols match). Pre:
  //   * if this matrix is empty (rows == 0 && cols == 0), adopts
  //     other's shape and data;
  //   * otherwise cols_ must equal other.cols_ (LOG-FATAL otherwise).
  // Post: rows_ becomes rows_ + other.rows_; cols_ unchanged; data_
  // gets other.data_ appended. Used by DoublePIR Setup to pad the
  // A2_copy matrix's rows up to a multiple of 3 before Transpose.
  //
  // Pure arithmetic — works in both stub and vendored modes.
  void Concat(const Matrix& other);

  // Squish / Unsquish — pure-arithmetic in-memory compression used by
  // the SimplePIR Answer path (matMulVecPacked). Pack `delta` adjacent
  // Z_p columns into one Z_q column where each Z_p value lives in a
  // `basis`-bit slot. Upstream simplepir uses basis=10, delta=3 so the
  // 32-bit Z_q column carries 3 Z_p slots of up to 1024 each.
  //
  // After Squish: rows unchanged, cols = ceil(cols / delta), and each
  // cell new[i, j] = sum_{k=0..delta-1, delta*j+k < old_cols}
  //                      old[i, delta*j+k] << (k * basis).
  //
  // Unsquish is the inverse — `orig_cols` is the un-squished column
  // count the caller stashed before calling Squish (Database mirrors
  // this in DBinfo.cols). After Unsquish: rows unchanged, cols =
  // orig_cols.
  //
  // Pure arithmetic — works in both stub and vendored modes. Database
  // wraps these to also update DBinfo.basis/squishing/cols so callers
  // pair the operations through the higher-level API.
  void Squish(uint64_t basis, uint64_t delta);
  void Unsquish(uint64_t basis, uint64_t delta, uint64_t orig_cols);

  // MulVecPacked — kernel bridge for the SimplePIR Answer path. `*this`
  // is a Squished matrix (L x C where C = ceil(M / squishing)). `b` is
  // a (C * squishing) x 1 column vector (the un-squished query, padded
  // with zeros if M is not divisible by squishing). Output is L x 1.
  //
  // The C kernel HARDCODES basis=10 and squishing=3 — passing other
  // values returns FAIL up-front. The kernel internally allocates an
  // (L+8) x 1 buffer to give itself room for SIMD tail iterations;
  // we DropLastRows(8) after the call to return the proper L x 1.
  retcode MulVecPacked(const Matrix& b, uint64_t basis,
                       uint64_t squishing, Matrix* out,
                       std::string* err) const;

  // Kernel bridges. Vendored mode forwards to upstream pir.c; stub
  // mode returns retcode::FAIL with the activation-flag guidance.
  //
  // MulVec: `*this` is the (a_rows x a_cols) matrix, `b` is an
  // a_cols x 1 column vector, `out` receives a_rows x 1.
  retcode MulVec(const Matrix& b, Matrix* out, std::string* err) const;

  // Mul: this (a_rows x a_cols) times b (a_cols x b_cols) -> out
  // (a_rows x b_cols). Wraps upstream matMul.
  retcode Mul(const Matrix& b, Matrix* out, std::string* err) const;

  // Transpose: writes a c_cols x r_rows matrix into `out`. In-place
  // variant is omitted — upstream's in-place Transpose() shuffles
  // pointers, which is harder to bound-check in C++; allocating a new
  // out is slightly more memory but lets the caller decide lifetime.
  retcode Transpose(Matrix* out, std::string* err) const;

 private:
  uint64_t rows_ = 0;
  uint64_t cols_ = 0;
  std::vector<uint32_t> data_;
};

}  // namespace primihub::pir::core

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_PIR_CORE_MATRIX_H_
