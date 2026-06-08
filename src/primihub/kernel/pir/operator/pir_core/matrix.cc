/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Matrix implementation. Bridge bodies for Mul / MulVec / Transpose
 * forward to the @simplepir C kernels under PIR_PIR_CORE_REAL.
 */
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>

#include <glog/logging.h>

#ifdef PIR_PIR_CORE_REAL
extern "C" {
typedef uint32_t Elem;
void matMul(Elem* out, const Elem* a, const Elem* b,
            size_t aRows, size_t aCols, size_t bCols);
void matMulVec(Elem* out, const Elem* a, const Elem* b,
               size_t aRows, size_t aCols);
void matMulVecPacked(Elem* out, const Elem* a, const Elem* b,
                     size_t aRows, size_t aCols);
void matMulTransposedPacked(Elem* out, const Elem* a, const Elem* b,
                            size_t aRows, size_t aCols,
                            size_t bRows, size_t bCols);
void transpose(Elem* out, const Elem* in, size_t rows, size_t cols);
}  // extern "C"
#endif  // PIR_PIR_CORE_REAL

namespace primihub::pir::core {

#ifdef PIR_PIR_CORE_REAL
const bool kPirCoreKernelsVendored = true;
#else
const bool kPirCoreKernelsVendored = false;
#endif

namespace {

void WriteNotVendoredError(std::string* err) {
  if (err) {
    *err =
        "primihub::pir::core::Matrix: kernels not vendored. Build with "
        "--define=enable_pir_core_real=1 and provide the @simplepir "
        "bazel override pointing at ahenzinger/simplepir (see "
        "docs/pir/activation-pattern.md).";
  }
}

}  // namespace

uint32_t Matrix::Get(uint64_t i, uint64_t j) const {
  if (i >= rows_ || j >= cols_) {
    LOG(FATAL) << "Matrix::Get out of bounds: (" << i << ", " << j
               << ") for shape (" << rows_ << ", " << cols_ << ")";
  }
  return data_[i * cols_ + j];
}

void Matrix::Set(uint64_t i, uint64_t j, uint32_t value) {
  if (i >= rows_ || j >= cols_) {
    LOG(FATAL) << "Matrix::Set out of bounds: (" << i << ", " << j
               << ") for shape (" << rows_ << ", " << cols_ << ")";
  }
  data_[i * cols_ + j] = value;
}

Matrix Matrix::UniformRandom(uint64_t rows, uint64_t cols, uint32_t logmod) {
  Matrix m(rows, cols);
  std::random_device rd;
  std::mt19937_64 gen(rd());
  // logmod == 32 means full uint32 range; smaller logmod truncates.
  const uint64_t bound = (logmod >= 32) ? (1ULL << 32) : (1ULL << logmod);
  std::uniform_int_distribution<uint64_t> dist(0, bound - 1);
  for (uint64_t i = 0; i < m.size(); ++i) {
    m.data_[i] = static_cast<uint32_t>(dist(gen));
  }
  return m;
}

void Matrix::MatrixAdd(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    LOG(FATAL) << "Matrix::MatrixAdd dim mismatch: " << rows_ << "x"
               << cols_ << " vs " << other.rows_ << "x" << other.cols_;
  }
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] += other.data_[i];
  }
}

void Matrix::MatrixSub(const Matrix& other) {
  if (rows_ != other.rows_ || cols_ != other.cols_) {
    LOG(FATAL) << "Matrix::MatrixSub dim mismatch: " << rows_ << "x"
               << cols_ << " vs " << other.rows_ << "x" << other.cols_;
  }
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] -= other.data_[i];
  }
}

void Matrix::ScalarAdd(uint32_t value) {
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] += value;
  }
}

void Matrix::ScalarSub(uint32_t value) {
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] -= value;
  }
}

void Matrix::ReduceMod(uint32_t modulus) {
  if (modulus == 0) {
    LOG(FATAL) << "Matrix::ReduceMod modulus == 0";
  }
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] %= modulus;
  }
}

void Matrix::DropLastRows(uint64_t n) {
  if (n > rows_) {
    LOG(FATAL) << "Matrix::DropLastRows n=" << n << " > rows_=" << rows_;
  }
  rows_ -= n;
  data_.resize(rows_ * cols_);
}

void Matrix::AppendZeros(uint64_t n) {
  if (cols_ != 1) {
    LOG(FATAL) << "Matrix::AppendZeros requires column vector (cols == 1) "
               << "but got cols_=" << cols_
               << "; upstream simplepir's AppendZeros calls Concat with a "
                  "(n, 1) zero matrix, which Concat rejects on cols mismatch";
  }
  if (n == 0) return;
  data_.resize(data_.size() + n, 0);
  rows_ += n;
}

void Matrix::ConcatCols(uint64_t n) {
  if (n == 1) return;
  if (n == 0) {
    LOG(FATAL) << "Matrix::ConcatCols n == 0";
  }
  if (cols_ % n != 0) {
    LOG(FATAL) << "Matrix::ConcatCols n=" << n
               << " does not divide cols_=" << cols_;
  }
  // Output is (rows_ * n) x (cols_ / n). For each input cell (i, j):
  //   new_col = j / n
  //   new_row = i + rows_ * (j % n)
  uint64_t new_rows = rows_ * n;
  uint64_t new_cols = cols_ / n;
  std::vector<uint32_t> out(new_rows * new_cols, 0);
  for (uint64_t i = 0; i < rows_; ++i) {
    for (uint64_t j = 0; j < cols_; ++j) {
      uint64_t nc = j / n;
      uint64_t nr = i + rows_ * (j % n);
      out[nr * new_cols + nc] = data_[i * cols_ + j];
    }
  }
  data_ = std::move(out);
  rows_ = new_rows;
  cols_ = new_cols;
}

void Matrix::Expand(uint64_t p, uint64_t delta) {
  if (p == 0) {
    LOG(FATAL) << "Matrix::Expand p == 0";
  }
  if (p == 1) {
    LOG(FATAL) << "Matrix::Expand p == 1 — every digit is 0";
  }
  if (delta == 0) {
    LOG(FATAL) << "Matrix::Expand delta == 0";
  }
  uint64_t new_rows = rows_ * delta;
  std::vector<uint32_t> out(new_rows * cols_, 0);
  uint32_t shift = static_cast<uint32_t>(p / 2);
  for (uint64_t i = 0; i < rows_; ++i) {
    for (uint64_t j = 0; j < cols_; ++j) {
      uint64_t val = data_[i * cols_ + j];
      for (uint64_t f = 0; f < delta; ++f) {
        uint32_t digit = static_cast<uint32_t>(val % p);
        // Centered representation: subtract p/2. Underflow on uint32
        // is intentional and matches upstream simplepir's centered
        // shift (Database::ScalarSub uses the same wrap-around).
        out[(i * delta + f) * cols_ + j] = digit - shift;
        val /= p;
      }
    }
  }
  data_ = std::move(out);
  rows_ = new_rows;
}

void Matrix::Contract(uint64_t p, uint64_t delta) {
  if (p == 0) {
    LOG(FATAL) << "Matrix::Contract p == 0";
  }
  if (delta == 0) {
    LOG(FATAL) << "Matrix::Contract delta == 0";
  }
  if (rows_ % delta != 0) {
    LOG(FATAL) << "Matrix::Contract delta=" << delta
               << " does not divide rows_=" << rows_;
  }
  uint64_t new_rows = rows_ / delta;
  std::vector<uint32_t> out(new_rows * cols_, 0);
  uint32_t shift = static_cast<uint32_t>(p / 2);
  for (uint64_t i = 0; i < new_rows; ++i) {
    for (uint64_t j = 0; j < cols_; ++j) {
      uint64_t acc = 0;
      uint64_t coeff = 1;
      for (uint64_t f = 0; f < delta; ++f) {
        uint32_t digit = data_[(i * delta + f) * cols_ + j];
        // Upstream verbatim: (uint64(digit) + p/2) % p. The uint64
        // widening before adding p/2 is exactly what upstream does;
        // it makes the underflow wrap from Expand visible as a large
        // uint64 (see the class-level docstring on round-trip
        // semantics).
        uint64_t recentered = (static_cast<uint64_t>(digit) + shift) % p;
        acc += coeff * recentered;
        coeff *= p;
      }
      out[i * cols_ + j] = static_cast<uint32_t>(acc);
    }
  }
  data_ = std::move(out);
  rows_ = new_rows;
}

void Matrix::Concat(const Matrix& other) {
  // Empty-target adopt path, mirrors upstream's first branch of
  // Concat which treats a zero-shape matrix as "uninitialized".
  if (rows_ == 0 && cols_ == 0) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = other.data_;
    return;
  }
  if (other.rows_ == 0) {
    // No-op: appending an empty row set leaves this unchanged.
    return;
  }
  if (cols_ != other.cols_) {
    LOG(FATAL) << "Matrix::Concat cols mismatch: this is " << rows_ << "x"
               << cols_ << ", other is " << other.rows_ << "x"
               << other.cols_;
  }
  data_.insert(data_.end(), other.data_.begin(), other.data_.end());
  rows_ += other.rows_;
}

void Matrix::Squish(uint64_t basis, uint64_t delta) {
  if (delta == 0) {
    LOG(FATAL) << "Matrix::Squish delta == 0";
  }
  if (basis == 0 || basis >= 32) {
    LOG(FATAL) << "Matrix::Squish basis=" << basis
               << " must be in (0, 32) — slot must fit in uint32 row";
  }
  const uint64_t new_cols = (cols_ + delta - 1) / delta;
  std::vector<uint32_t> packed(rows_ * new_cols, 0);
  for (uint64_t i = 0; i < rows_; ++i) {
    for (uint64_t j = 0; j < new_cols; ++j) {
      uint32_t acc = 0;
      for (uint64_t k = 0; k < delta; ++k) {
        const uint64_t src_col = delta * j + k;
        if (src_col >= cols_) break;
        const uint32_t val = data_[i * cols_ + src_col];
        acc += val << (k * basis);
      }
      packed[i * new_cols + j] = acc;
    }
  }
  cols_ = new_cols;
  data_ = std::move(packed);
}

void Matrix::Unsquish(uint64_t basis, uint64_t delta, uint64_t orig_cols) {
  if (delta == 0) {
    LOG(FATAL) << "Matrix::Unsquish delta == 0";
  }
  if (basis == 0 || basis >= 32) {
    LOG(FATAL) << "Matrix::Unsquish basis=" << basis
               << " must be in (0, 32)";
  }
  // Sanity: cols_ should equal ceil(orig_cols / delta) if this matrix
  // really came from Squish(basis, delta). Off-by-one here would write
  // garbage; LOG(FATAL) so the caller sees the mismatch immediately.
  const uint64_t expected_packed_cols = (orig_cols + delta - 1) / delta;
  if (cols_ != expected_packed_cols) {
    LOG(FATAL) << "Matrix::Unsquish cols mismatch — current cols=" << cols_
               << " but orig_cols=" << orig_cols << " with delta=" << delta
               << " implies packed cols=" << expected_packed_cols
               << ". Caller must pass the same orig_cols stashed before"
               << " Squish.";
  }
  const uint64_t mask = (1ULL << basis) - 1;
  std::vector<uint32_t> unpacked(rows_ * orig_cols, 0);
  for (uint64_t i = 0; i < rows_; ++i) {
    for (uint64_t j = 0; j < cols_; ++j) {
      const uint32_t packed_val = data_[i * cols_ + j];
      for (uint64_t k = 0; k < delta; ++k) {
        const uint64_t dst_col = j * delta + k;
        if (dst_col >= orig_cols) break;
        unpacked[i * orig_cols + dst_col] =
            static_cast<uint32_t>((packed_val >> (k * basis)) & mask);
      }
    }
  }
  cols_ = orig_cols;
  data_ = std::move(unpacked);
}

retcode Matrix::MulVec(const Matrix& b, Matrix* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Matrix::MulVec: out is null";
    return retcode::FAIL;
  }
  if (b.cols_ != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulVec: b must be a column vector; got cols=" << b.cols_;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (cols_ != b.rows_) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulVec: dim mismatch " << rows_ << "x" << cols_
          << " * " << b.rows_ << "x" << b.cols_;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
#ifndef PIR_PIR_CORE_REAL
  WriteNotVendoredError(err);
  return retcode::FAIL;
#else
  *out = Matrix(rows_, 1);
  matMulVec(out->mutable_data(), data_.data(), b.data(),
            static_cast<size_t>(rows_), static_cast<size_t>(cols_));
  return retcode::SUCCESS;
#endif
}

retcode Matrix::Mul(const Matrix& b, Matrix* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Matrix::Mul: out is null";
    return retcode::FAIL;
  }
  if (b.cols_ == 1) {
    return MulVec(b, out, err);
  }
  if (cols_ != b.rows_) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::Mul: dim mismatch " << rows_ << "x" << cols_
          << " * " << b.rows_ << "x" << b.cols_;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
#ifndef PIR_PIR_CORE_REAL
  WriteNotVendoredError(err);
  return retcode::FAIL;
#else
  *out = Matrix(rows_, b.cols_);
  matMul(out->mutable_data(), data_.data(), b.data(),
         static_cast<size_t>(rows_), static_cast<size_t>(cols_),
         static_cast<size_t>(b.cols_));
  return retcode::SUCCESS;
#endif
}

retcode Matrix::MulVecPacked(const Matrix& b, uint64_t basis,
                             uint64_t squishing, Matrix* out,
                             std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Matrix::MulVecPacked: out is null";
    return retcode::FAIL;
  }
  // Upstream pir.c hardcodes basis=10 / compression=3 for the SIMD
  // shift constants. Reject anything else up-front so callers do not
  // get silently-wrong answers.
  if (basis != 10 || squishing != 3) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulVecPacked: basis=" << basis
          << " squishing=" << squishing
          << " — kernel hardcodes basis=10 squishing=3. Pass those"
          << " literals or wait for a future kernel that parameterizes.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (b.cols_ != 1) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulVecPacked: b must be a column vector; got cols="
          << b.cols_;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Upstream MatrixMulVecPacked requires `a.Cols * compression == b.Rows`,
  // i.e. the unpacked query length matches the packed-DB column count
  // times the compression factor. Surface that as a guard.
  if (cols_ * squishing != b.rows_) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulVecPacked: dim mismatch — packed DB is "
          << rows_ << "x" << cols_
          << " (expects b.rows = cols * squishing = "
          << (cols_ * squishing) << "); got b="
          << b.rows_ << "x" << b.cols_ << ".";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
#ifndef PIR_PIR_CORE_REAL
  WriteNotVendoredError(err);
  return retcode::FAIL;
#else
  // Kernel writes rows_+8 elements (SIMD tail padding); allocate
  // accordingly then DropLastRows(8) to surface the proper L x 1.
  Matrix temp(rows_ + 8, 1);
  matMulVecPacked(temp.mutable_data(), data_.data(), b.data(),
                  static_cast<size_t>(rows_),
                  static_cast<size_t>(cols_));
  temp.DropLastRows(8);
  *out = std::move(temp);
  return retcode::SUCCESS;
#endif
}

retcode Matrix::Transpose(Matrix* out, std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Matrix::Transpose: out is null";
    return retcode::FAIL;
  }
#ifndef PIR_PIR_CORE_REAL
  WriteNotVendoredError(err);
  return retcode::FAIL;
#else
  *out = Matrix(cols_, rows_);
  transpose(out->mutable_data(), data_.data(),
            static_cast<size_t>(rows_), static_cast<size_t>(cols_));
  return retcode::SUCCESS;
#endif
}


Matrix Matrix::SelectRows(uint64_t offset, uint64_t num_rows) const {
  if (offset + num_rows > rows_) {
    LOG(FATAL) << "Matrix::SelectRows offset=" << offset << " num_rows="
               << num_rows << " > rows_=" << rows_;
  }
  Matrix out(num_rows, cols_);
  if (num_rows == 0) {
    return out;
  }
  std::copy(data_.begin() + offset * cols_,
            data_.begin() + (offset + num_rows) * cols_,
            out.data_.begin());
  return out;
}

void Matrix::Round(const LweParams& params) {
  // LweParams::Round handles the per-cell math (Delta + p semantics).
  // Iterating element-by-element mirrors upstream simplepir matrix.go
  // Matrix.Round.
  for (uint64_t i = 0; i < size(); ++i) {
    data_[i] = static_cast<uint32_t>(
        params.Round(static_cast<uint64_t>(data_[i])));
  }
}

retcode Matrix::MulTransposedPacked(const Matrix& b, uint64_t basis,
                                     uint64_t squishing, Matrix* out,
                                     std::string* err) const {
  if (out == nullptr) {
    if (err) *err = "Matrix::MulTransposedPacked: out is null";
    return retcode::FAIL;
  }
  if (basis != 10 || squishing != 3) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulTransposedPacked: basis=" << basis
          << " squishing=" << squishing
          << " — kernel hardcodes basis=10 squishing=3.";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Upstream kernel layout: `this` is squished (cols = packed cols);
  // `b` is the un-squished operand whose b.cols equals this.cols *
  // squishing (the kernel reads b at offset k*COMPRESSION+j*bCols
  // for k in [0, this.cols) — needing b.cols >= this.cols * 3 to
  // stay in-bounds). Match upstream exactly: require equality.
  if (b.cols_ != cols_ * squishing) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulTransposedPacked: b.cols must equal "
          << "this.cols * squishing — this=" << rows_ << "x" << cols_
          << ", b=" << b.rows_ << "x" << b.cols_
          << ", squishing=" << squishing
          << " (kernel reads b at k*COMPRESSION+j*bCols up to "
          << "k=this.cols-1).";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  // Short-rows branch (aRows <= aCols) steps j by 8 — bRows must
  // be a multiple of 8 in that branch. Long-rows branch (aRows >
  // aCols) steps j by 1 and tolerates any bRows. Match upstream by
  // gating the alignment check on the branch the kernel will take.
  if (rows_ <= cols_ && (b.rows_ % 8) != 0) {
    if (err) {
      std::ostringstream oss;
      oss << "Matrix::MulTransposedPacked: b.rows=" << b.rows_
          << " must be a multiple of 8 when this.rows <= this.cols "
          << "(kernel short-rows branch unrolls the j-loop by 8). "
          << "this=" << rows_ << "x" << cols_ << ", b=" << b.rows_
          << "x" << b.cols_ << ".";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
#ifndef PIR_PIR_CORE_REAL
  WriteNotVendoredError(err);
  return retcode::FAIL;
#else
  // Output shape: (this.rows) x (b.rows). Both operands packed.
  Matrix temp(rows_, b.rows_);
  // The C kernel reads aRows, aCols, bRows, bCols and writes
  // out[bRows*i + j]. Output is aRows-by-bRows in row-major.
  matMulTransposedPacked(temp.mutable_data(), data_.data(), b.data(),
                          static_cast<size_t>(rows_),
                          static_cast<size_t>(cols_),
                          static_cast<size_t>(b.rows_),
                          static_cast<size_t>(b.cols_));
  *out = std::move(temp);
  return retcode::SUCCESS;
#endif
}

}  // namespace primihub::pir::core
