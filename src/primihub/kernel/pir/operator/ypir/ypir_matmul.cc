/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_matmul.h"

#include <sstream>

#ifdef PIR_YPIR_RUNTIME_VENDORED
extern "C" {
// Forward declarations mirror upstream src/matmul.cpp `extern "C"`
// block. Upstream has no public header for these symbols. Keeping
// the declarations local avoids a drifting wrapper header.
void matMulVecPacked(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     std::size_t a_rows, std::size_t a_cols);
void matMulVecPacked2(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      std::size_t a_rows, std::size_t a_cols);
void matMulVecPacked4(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      std::size_t a_rows, std::size_t a_cols);
void matMulVecPacked8(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      std::size_t a_rows, std::size_t a_cols);
}  // extern "C"
#endif  // PIR_YPIR_RUNTIME_VENDORED

namespace primihub::pir::ypir {

#ifdef PIR_YPIR_RUNTIME_VENDORED
const bool kYpirMatmulVendored = true;
#else
const bool kYpirMatmulVendored = false;
#endif

namespace {

inline void SetErr(std::string* err, std::string msg) {
  if (err) {
    *err = std::move(msg);
  }
}

}  // namespace

retcode MatMulVecPacked(uint32_t* out, std::size_t out_len,
                        const uint32_t* a, std::size_t a_len,
                        const uint32_t* b, std::size_t b_len,
                        std::size_t a_rows, std::size_t a_cols,
                        std::size_t b_rows, std::size_t b_cols,
                        std::string* err) {
  if (out == nullptr || a == nullptr || b == nullptr) {
    SetErr(err,
           "YpirMatMulVecPacked: out / a / b must all be non-null");
    return retcode::FAIL;
  }
  if (a_len != a_rows * a_cols) {
    std::ostringstream oss;
    oss << "YpirMatMulVecPacked: a_len (" << a_len
        << ") != a_rows * a_cols (" << a_rows << " * " << a_cols
        << " = " << (a_rows * a_cols) << ")";
    SetErr(err, oss.str());
    return retcode::FAIL;
  }
  if (b_len != b_rows * b_cols) {
    std::ostringstream oss;
    oss << "YpirMatMulVecPacked: b_len (" << b_len
        << ") != b_rows * b_cols (" << b_rows << " * " << b_cols
        << " = " << (b_rows * b_cols) << ")";
    SetErr(err, oss.str());
    return retcode::FAIL;
  }
  if (a_cols * 4 != b_rows) {
    std::ostringstream oss;
    oss << "YpirMatMulVecPacked: packed-base shape contract "
        << "violated — a_cols * 4 (" << a_cols << " * 4 = "
        << (a_cols * 4) << ") != b_rows (" << b_rows
        << "). Upstream menonsamir/ypir matmul.rs requires "
        << "COMPRESSION=4 packed bytes per a-entry.";
    SetErr(err, oss.str());
    return retcode::FAIL;
  }
  // SIMD tail slack: upstream test test_matmul_vec_packed_8 allocates
  // `(a_rows + 8) * b_cols` for the output buffer.
  const std::size_t required_out = (a_rows + 8u) * b_cols;
  if (out_len < required_out) {
    std::ostringstream oss;
    oss << "YpirMatMulVecPacked: out_len (" << out_len
        << ") < required (a_rows + 8) * b_cols ("
        << "(" << a_rows << " + 8) * " << b_cols << " = "
        << required_out << "). Required for SIMD tail-write slack.";
    SetErr(err, oss.str());
    return retcode::FAIL;
  }
#ifndef PIR_YPIR_RUNTIME_VENDORED
  SetErr(err,
         "YpirMatMulVecPacked: not vendored. Build with "
         "--define=enable_ypir_real=1 and provide the @ypir bazel "
         "override (menonsamir/ypir@a73e550a) to link the matmul "
         "kernels. See openspec task 7.3 chunk 2 in "
         "docs/pir/ypir-port-plan.md.");
  return retcode::FAIL;
#else
  switch (b_cols) {
    case 1:
      matMulVecPacked(out, a, b, a_rows, a_cols);
      return retcode::SUCCESS;
    case 2:
      matMulVecPacked2(out, a, b, a_rows, a_cols);
      return retcode::SUCCESS;
    case 4:
      matMulVecPacked4(out, a, b, a_rows, a_cols);
      return retcode::SUCCESS;
    case 8:
      matMulVecPacked8(out, a, b, a_rows, a_cols);
      return retcode::SUCCESS;
    default: {
      std::ostringstream oss;
      oss << "YpirMatMulVecPacked: b_cols must be in {1, 2, 4, 8}, "
          << "got " << b_cols
          << ". Upstream matmul.rs panics on this; we return FAIL "
          << "to preserve caller error handling. b_cols=6 is "
          << "intentionally disabled (commented out upstream).";
      SetErr(err, oss.str());
      return retcode::FAIL;
    }
  }
#endif  // PIR_YPIR_RUNTIME_VENDORED
}

}  // namespace primihub::pir::ypir
