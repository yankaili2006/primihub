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
