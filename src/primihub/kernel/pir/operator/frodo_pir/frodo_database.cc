/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_database.h"

#include <cassert>
#include <utility>

#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_matrices.h"

namespace primihub::pir::frodo {

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

}  // namespace primihub::pir::frodo
