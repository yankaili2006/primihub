/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_transpose — C++ port of upstream menonsamir/ypir@a73e550a
 * src/transpose.rs. Pure byte / element / f64 transpose helpers
 * with no algorithmic dependency on the rest of YPIR — first chunk
 * of the YPIR port (see docs/pir/ypir-port-plan.md).
 *
 * Three flavours mirror the Rust:
 *   * Transpose(buf, rows, cols, bytes_per_pt_el) — byte-level,
 *     used by packing.rs when reshaping serialized payloads.
 *   * TransposeElems<T>(rows of T) — element-level for non-trivially-
 *     copyable T, used by client.rs reshaping the hint.
 *   * TransposeGeneric<T>(buf, rows, cols) — 32×32 tiled when sizes
 *     divide, falls back to 1×1 otherwise; used by server.rs hint
 *     reshape on the hot path.
 *   * TransposeF64(out, in, rows, cols) — same as Generic but with
 *     a fixed 32-tile and no fallback; matches the upstream
 *     `transpose_f64` exactly (asserts on size). Used by f64 path.
 *
 * NOT vendored-gated: these are upstream-port code, present in
 * every build configuration. The @ypir override only matters for
 * the matmul kernels (ypir_runtime.cc).
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_TRANSPOSE_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_TRANSPOSE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace primihub::pir::ypir {

// Byte-level transpose. `buf` is `rows * cols * bytes_per_pt_el`
// bytes laid out row-major; output is the same buffer transposed so
// that element (j, i) ends up at offset `(j * rows + i) * bytes_per_pt_el`.
//
// Mirrors `transpose(buf, rows, cols, bytes_per_pt_el)` in upstream
// transpose.rs. Returns a freshly-allocated vector; callers that want
// to write into an existing buffer can copy out.
std::vector<uint8_t> Transpose(const uint8_t* buf, std::size_t buf_len,
                               std::size_t rows, std::size_t cols,
                               std::size_t bytes_per_pt_el);

// Element-level transpose for non-trivially-copyable T. Input is an
// `inp_rows` × `inp_cols` matrix represented as a vector of row
// vectors (each row has `inp_cols` entries). Output is `inp_cols`
// rows of `inp_rows` entries each. Mirrors upstream `transpose_elems`.
template <typename T>
std::vector<std::vector<T>> TransposeElems(
    const std::vector<std::vector<T>>& buf, std::size_t inp_rows,
    std::size_t inp_cols) {
  std::vector<std::vector<T>> out;
  out.reserve(inp_cols);
  for (std::size_t j = 0; j < inp_cols; ++j) {
    std::vector<T> row;
    row.reserve(inp_rows);
    for (std::size_t i = 0; i < inp_rows; ++i) {
      row.push_back(buf[i][j]);
    }
    out.push_back(std::move(row));
  }
  return out;
}

// Generic tiled transpose. Tile size is 32 when both dims are a
// multiple of 32, otherwise falls back to 1 (= the naïve loop).
// Mirrors upstream `transpose_generic`. `T` must be default-
// constructible and trivially assignable.
template <typename T>
std::vector<T> TransposeGeneric(const std::vector<T>& a,
                                std::size_t a_rows, std::size_t a_cols) {
  std::size_t tile = 32;
  if (tile > a_rows) tile = a_rows;
  if (tile > a_cols) tile = a_cols;
  if (tile == 0 || a_rows % tile != 0 || a_cols % tile != 0) {
    tile = 1;
  }
  std::vector<T> out(a_rows * a_cols);
  for (std::size_t i_outer = 0; i_outer < a_rows; i_outer += tile) {
    for (std::size_t j_outer = 0; j_outer < a_cols; j_outer += tile) {
      for (std::size_t i_inner = 0; i_inner < tile; ++i_inner) {
        for (std::size_t j_inner = 0; j_inner < tile; ++j_inner) {
          std::size_t i = i_outer + i_inner;
          std::size_t j = j_outer + j_inner;
          out[j * a_rows + i] = a[i * a_cols + j];
        }
      }
    }
  }
  return out;
}

// Fixed-tile (32) f64 transpose into a caller-owned buffer. Asserts
// (via runtime check returning early-zero output via debug-only
// abort path is intentionally avoided; signal misuse with size==0
// outputs is the C++ choice here) on `a_rows` and `a_cols` not being
// >= 32 and not being multiples of 32. Matches upstream
// `transpose_f64(out, a, a_rows, a_cols)` exactly when the size
// preconditions hold.
//
// Returns true on success, false if preconditions are violated; in
// the failure case `out` is left untouched.
bool TransposeF64(double* out, const double* a, std::size_t a_rows,
                  std::size_t a_cols);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_TRANSPOSE_H_
