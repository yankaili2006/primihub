/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <utility>

#include "base64.h"  // NOLINT — @com_github_base64_cpp//:base64_lib

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_format.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

namespace primihub::pir::frodo {

Database::Database() : m_(0), elem_size_(0), plaintext_bits_(0) {}

// chunk g-4: data ctor accepts the legacy nested vec<vec<u32>>
// layout (column-form, entries[c][r]) and materialises it into a
// ColMajorMatrix of identical shape (height = entries[0].size(),
// width = entries.size(), col c stored at storage[c*height ..
// c*height + height]). Tests that already build nested fixtures
// keep compiling unchanged; the production-time SwitchFmt /
// VecMult / GetRow / GetDbEntry callers all see the flat backing
// from this point forward.
Database::Database(std::vector<std::vector<std::uint32_t>> entries,
                   std::size_t m, std::size_t elem_size,
                   std::size_t plaintext_bits)
    : m_(m),
      elem_size_(elem_size),
      plaintext_bits_(plaintext_bits) {
  const std::size_t width = entries.size();
  const std::size_t height = width == 0 ? 0 : entries[0].size();
  entries_ = ColMajorMatrix(height, width, ColMajorMatrix::NoInit{});
  for (std::size_t c = 0; c < width; ++c) {
    // Tests pass uniform-width columns; copy each into the
    // contiguous height-sized run for column c.
    assert(entries[c].size() == height &&
           "Database ctor: non-uniform column heights");
    std::uint32_t* dst = entries_.column_data(c);
    const std::vector<std::uint32_t>& src = entries[c];
    const std::size_t n = std::min(height, src.size());
    for (std::size_t r = 0; r < n; ++r) {
      dst[r] = src[r];
    }
  }
}

void Database::SwitchFmt() {
  // chunk g-4: route through the flat-buffer overload landed in
  // chunk g-3. No nested-vector materialisation: previously this
  // line was `entries_ = SwapMatrixFmt(entries_)`, paying ~5.7 s
  // at N=1M for the per-column alloc + value-init dance.
  entries_ = SwapMatrixFmtFlat(entries_);
}

std::uint32_t Database::VecMult(const std::vector<std::uint32_t>& row,
                                std::size_t col_idx) const {
  // Upstream: vec_mult_u32_u32(row, &self.entries[col_idx]).unwrap()
  // .unwrap() panics on size mismatch — we treat that as a caller
  // contract violation and return 0 in release (assertion in debug).
  assert(col_idx < entries_.width() && "VecMult: col_idx OOB");
  if (col_idx >= entries_.width()) {
    return 0u;
  }
  // chunk g-4: feed the raw column pointer + height directly to the
  // raw-pointer VecMultU32U32 overload. Previously this materialised
  // a nested entries_[col_idx] vector for every call — at N=1M,
  // dim=512 the temporary was 4 MB per VecMult.
  std::uint32_t out = 0u;
  std::string err;
  const auto rc = VecMultU32U32(row, entries_.column_data(col_idx),
                                entries_.height(), &out, &err);
  assert(rc == retcode::SUCCESS && "VecMult: size mismatch");
  if (rc != retcode::SUCCESS) {
    return 0u;
  }
  return out;
}

std::uint32_t Database::VecMult(const std::uint32_t* row,
                                std::size_t row_len,
                                std::size_t col_idx) const {
  // chunk g-5 follow-up: zero-overhead VecMult for hot loops where
  // both operands already live in flat backings. Caller has verified
  // shapes; no err propagation or diagnostic allocation.
  assert(col_idx < entries_.width() && "VecMult: col_idx OOB");
  assert(row_len == entries_.height() && "VecMult: row_len mismatch");
  if (col_idx >= entries_.width() || row_len != entries_.height()) {
    return 0u;
  }
  std::uint32_t out = 0u;
  VecMultU32U32(row, entries_.column_data(col_idx), row_len, &out);
  return out;
}

std::vector<std::uint32_t> Database::GetRow(std::size_t i) const {
  // chunk g-4: i indexes a column in the column-major matrix
  // (preserving upstream's `entries[i]` semantic). OOB returns
  // empty per the soft-boundary convention.
  if (i >= entries_.width()) {
    return {};
  }
  const std::uint32_t* col = entries_.column_data(i);
  return std::vector<std::uint32_t>(col, col + entries_.height());
}

std::size_t Database::GetMatrixWidth(std::size_t elem_size,
                                     std::size_t plaintext_bits) {
  if (plaintext_bits == 0) {
    return 0;
  }
  std::size_t quo = elem_size / plaintext_bits;
  if (elem_size % plaintext_bits != 0) {
    quo += 1;
  }
  return quo;
}

std::size_t Database::GetMatrixWidthSelf() const {
  return GetMatrixWidth(elem_size_, plaintext_bits_);
}

retcode ConstructRows(const std::vector<std::string>& elements,
                      std::size_t m, std::size_t elem_size,
                      std::size_t plaintext_bits,
                      std::vector<std::vector<std::uint32_t>>* out,
                      std::string* err) {
  if (out == nullptr) {
    if (err) *err = "ConstructRows: out must be non-null";
    return retcode::FAIL;
  }
  if (elements.size() != m) {
    if (err) {
      std::ostringstream oss;
      oss << "ConstructRows: elements.size()=" << elements.size()
          << " != m=" << m
          << ". Upstream would index OOB at elements[i].";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  const std::size_t row_width =
      Database::GetMatrixWidth(elem_size, plaintext_bits);
  std::vector<std::vector<std::uint32_t>> result;
  result.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    // Upstream: bytes = base64::decode(elements[i]); bits =
    // bytes_to_bits_le(&bytes); then per-row_width chunks call
    // bits_to_u32_le(&bits[i * plaintext_bits .. end_bound]).
    const std::string& b64 = elements[i];
    const std::string bytes = base64_decode(b64);
    if (!b64.empty() && bytes.empty()) {
      if (err) {
        std::ostringstream oss;
        oss << "ConstructRows: base64 decode failed at elements["
            << i << "] (non-empty input decoded to empty bytes).";
        *err = oss.str();
      }
      return retcode::FAIL;
    }
    std::vector<std::uint8_t> bytes_vec(bytes.begin(), bytes.end());
    auto bits = BytesToBitsLe(bytes_vec);

    std::vector<std::uint32_t> row;
    row.reserve(row_width);
    for (std::size_t j = 0; j < row_width; ++j) {
      const std::size_t start = j * plaintext_bits;
      const std::size_t end_bound = (j + 1) * plaintext_bits;
      std::vector<std::uint8_t> chunk;
      if (end_bound < bits.size()) {
        chunk.assign(bits.begin() + start, bits.begin() + end_bound);
      } else if (start <= bits.size()) {
        chunk.assign(bits.begin() + start, bits.end());
      } else {
        chunk.clear();
      }
      std::uint32_t v = 0;
      const auto rc = BitsToU32Le(chunk, &v, err);
      if (rc != retcode::SUCCESS) {
        if (err) {
          std::ostringstream oss;
          oss << "ConstructRows: BitsToU32Le failed at elements["
              << i << "] chunk " << j << ": " << *err;
          *err = oss.str();
        }
        return retcode::FAIL;
      }
      row.push_back(v);
    }
    result.push_back(std::move(row));
  }
  *out = std::move(result);
  return retcode::SUCCESS;
}

retcode Database::New(const std::vector<std::string>& elements,
                      std::size_t m, std::size_t elem_size,
                      std::size_t plaintext_bits, Database* out_db,
                      std::string* err) {
  if (out_db == nullptr) {
    if (err) *err = "Database::New: out_db must be non-null";
    return retcode::FAIL;
  }
  std::vector<std::vector<std::uint32_t>> rows;
  const auto rc = ConstructRows(elements, m, elem_size,
                                plaintext_bits, &rows, err);
  if (rc != retcode::SUCCESS) {
    return rc;
  }
  // Upstream: entries = swap_matrix_fmt(&construct_rows(...)).
  // The convention throughout db.rs is that Database starts in
  // column-form.
  //
  // chunk g-4: ConstructRows still produces a nested vec<vec<u32>>
  // (row-form, m rows by row_width). To land the flat column-form
  // we (1) lift the row-form into a ColMajorMatrix that mirrors
  // its layout (height=row_width, width=m, treating each row of
  // `rows` as a column of the ColMajorMatrix), then (2) call
  // SwapMatrixFmtFlat to produce the column-form (height=m,
  // width=row_width). The end result is byte-for-byte equivalent
  // to `SwapMatrixFmt(rows)` followed by the chunk g-4 ctor copy.
  // ConstructRows itself is migrated in chunk g-6 once all
  // callers reach ColMajorMatrix directly.
  const std::size_t row_width =
      Database::GetMatrixWidth(elem_size, plaintext_bits);
  ColMajorMatrix row_form(/*height=*/row_width, /*width=*/m,
                          ColMajorMatrix::NoInit{});
  for (std::size_t i = 0; i < m; ++i) {
    std::uint32_t* col = row_form.column_data(i);
    const std::vector<std::uint32_t>& src = rows[i];
    const std::size_t n = std::min(row_width, src.size());
    for (std::size_t j = 0; j < n; ++j) {
      col[j] = src[j];
    }
  }
  out_db->entries_ = SwapMatrixFmtFlat(row_form);
  out_db->m_ = m;
  out_db->elem_size_ = elem_size;
  out_db->plaintext_bits_ = plaintext_bits;
  return retcode::SUCCESS;
}



std::string Database::GetDbEntry(std::size_t i) const {
  // Upstream: base64_from_u32_slice(
  //              &get_matrix_second_at(&self.entries, i),
  //              self.plaintext_bits, self.elem_size).
  // GetMatrixSecondAt returns empty on OOB or empty input — that
  // propagates through Base64FromU32Slice which also returns empty
  // on empty input, so the OOB soft boundary is honored end-to-end.
  // chunk g-4: entries_ is now a ColMajorMatrix; call the
  // GetMatrixSecondAtFlat sibling that takes a ColMajorMatrix.
  return Base64FromU32Slice(
      GetMatrixSecondAtFlat(entries_, i), plaintext_bits_, elem_size_);
}

}  // namespace primihub::pir::frodo
