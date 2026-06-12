/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"

#include <cassert>
#include <sstream>
#include <utility>

#include "base64.h"  // NOLINT — @com_github_base64_cpp//:base64_lib

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_format.h"
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

namespace primihub::pir::frodo {

Database::Database() : m_(0), elem_size_(0), plaintext_bits_(0) {}

Database::Database(std::vector<std::vector<std::uint32_t>> entries,
                   std::size_t m, std::size_t elem_size,
                   std::size_t plaintext_bits)
    : entries_(std::move(entries)),
      m_(m),
      elem_size_(elem_size),
      plaintext_bits_(plaintext_bits) {}

void Database::SwitchFmt() {
  entries_ = SwapMatrixFmt(entries_);
}

std::uint32_t Database::VecMult(const std::vector<std::uint32_t>& row,
                                std::size_t col_idx) const {
  // Upstream: vec_mult_u32_u32(row, &self.entries[col_idx]).unwrap()
  // .unwrap() panics on size mismatch — we treat that as a caller
  // contract violation and return 0 in release (assertion in debug).
  assert(col_idx < entries_.size() && "VecMult: col_idx OOB");
  if (col_idx >= entries_.size()) {
    return 0u;
  }
  std::uint32_t out = 0u;
  std::string err;
  const auto rc = VecMultU32U32(row, entries_[col_idx], &out, &err);
  assert(rc == retcode::SUCCESS && "VecMult: size mismatch");
  if (rc != retcode::SUCCESS) {
    return 0u;
  }
  return out;
}

std::vector<std::uint32_t> Database::GetRow(std::size_t i) const {
  if (i >= entries_.size()) {
    return {};
  }
  return entries_[i];
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
  out_db->entries_ = SwapMatrixFmt(rows);
  out_db->m_ = m;
  out_db->elem_size_ = elem_size;
  out_db->plaintext_bits_ = plaintext_bits;
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::frodo
