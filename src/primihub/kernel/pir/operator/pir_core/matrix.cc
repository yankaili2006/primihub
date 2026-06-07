/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Matrix implementation. Bridge bodies for Mul / MulVec / Transpose
 * forward to the @simplepir C kernels under PIR_PIR_CORE_REAL.
 */
#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

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

}  // namespace primihub::pir::core
