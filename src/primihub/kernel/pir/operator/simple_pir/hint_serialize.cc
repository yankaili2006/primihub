/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/simple_pir/hint_serialize.h"

#include <cstdint>
#include <cstring>
#include <sstream>

#include <glog/logging.h>

namespace primihub::pir::simple_pir {

namespace {

constexpr uint64_t kMaxMatrixCells = static_cast<uint64_t>(1) << 32;

inline void WriteU16(uint16_t v, std::string* out) {
  uint8_t buf[2] = {static_cast<uint8_t>(v & 0xff),
                    static_cast<uint8_t>((v >> 8) & 0xff)};
  out->append(reinterpret_cast<const char*>(buf), 2);
}

inline void WriteU32(uint32_t v, std::string* out) {
  uint8_t buf[4];
  for (int i = 0; i < 4; ++i) {
    buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
  }
  out->append(reinterpret_cast<const char*>(buf), 4);
}

inline void WriteU64(uint64_t v, std::string* out) {
  uint8_t buf[8];
  for (int i = 0; i < 8; ++i) {
    buf[i] = static_cast<uint8_t>((v >> (i * 8)) & 0xff);
  }
  out->append(reinterpret_cast<const char*>(buf), 8);
}

inline bool ReadU16(const std::string& blob, size_t* off, uint16_t* out) {
  if (*off + 2 > blob.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(blob.data() + *off);
  *out = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
  *off += 2;
  return true;
}

inline bool ReadU32(const std::string& blob, size_t* off, uint32_t* out) {
  if (*off + 4 > blob.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(blob.data() + *off);
  *out = 0;
  for (int i = 0; i < 4; ++i) {
    *out |= static_cast<uint32_t>(p[i]) << (i * 8);
  }
  *off += 4;
  return true;
}

inline bool ReadU64(const std::string& blob, size_t* off, uint64_t* out) {
  if (*off + 8 > blob.size()) return false;
  const uint8_t* p = reinterpret_cast<const uint8_t*>(blob.data() + *off);
  *out = 0;
  for (int i = 0; i < 8; ++i) {
    *out |= static_cast<uint64_t>(p[i]) << (i * 8);
  }
  *off += 8;
  return true;
}

void WriteMatrix(const core::Matrix& m, std::string* out) {
  WriteU64(m.rows(), out);
  WriteU64(m.cols(), out);
  const uint32_t* src = m.data();
  const uint64_t cells = m.size();
  out->reserve(out->size() + cells * 4);
  for (uint64_t k = 0; k < cells; ++k) {
    WriteU32(src[k], out);
  }
}

retcode ReadMatrix(const std::string& blob, size_t* off, core::Matrix* m,
                   std::string* err, const char* matrix_name) {
  uint64_t rows = 0, cols = 0;
  if (!ReadU64(blob, off, &rows) || !ReadU64(blob, off, &cols)) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: truncated header for matrix "
          << matrix_name;
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (cols != 0 && rows > kMaxMatrixCells / cols) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: matrix " << matrix_name
          << " rows*cols overflow (" << rows << " * " << cols << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  const uint64_t cells = rows * cols;
  if (cells > kMaxMatrixCells) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: matrix " << matrix_name
          << " exceeds kMaxMatrixCells (" << cells << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (*off + cells * 4 > blob.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: truncated body for matrix "
          << matrix_name << " (need " << cells * 4 << " bytes, have "
          << (blob.size() - *off) << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  *m = core::Matrix::Zeros(rows, cols);
  uint32_t* dst = m->mutable_data();
  for (uint64_t k = 0; k < cells; ++k) {
    uint32_t v = 0;
    if (!ReadU32(blob, off, &v)) {
      if (err) *err = "simple_pir DeserializeHint: unexpected EOF mid-matrix";
      return retcode::FAIL;
    }
    dst[k] = v;
  }
  return retcode::SUCCESS;
}

}  // namespace

retcode SerializeHint(const SimplePirHint& hint, std::string* blob_out,
                      std::string* err) {
  if (blob_out == nullptr) {
    if (err) *err = "simple_pir SerializeHint: blob_out is null";
    return retcode::FAIL;
  }
  blob_out->clear();
  const uint64_t estimate = 120 + 4 * (hint.A.size() + hint.H.size());
  blob_out->reserve(estimate);

  blob_out->append(kSimpleHintMagic, 4);
  WriteU16(kSimpleHintWireVersion, blob_out);
  WriteU16(0, blob_out);
  const auto& info = hint.info_after_squish;
  WriteU64(info.num, blob_out);
  WriteU64(info.row_length, blob_out);
  WriteU64(info.packing, blob_out);
  WriteU64(info.ne, blob_out);
  WriteU64(info.x, blob_out);
  WriteU64(info.p, blob_out);
  WriteU64(info.logq, blob_out);
  WriteU64(info.basis, blob_out);
  WriteU64(info.squishing, blob_out);
  WriteU64(info.cols, blob_out);

  WriteMatrix(hint.A, blob_out);
  WriteMatrix(hint.H, blob_out);
  return retcode::SUCCESS;
}

retcode DeserializeHint(const std::string& blob, SimplePirHint* hint_out,
                        std::string* err) {
  if (hint_out == nullptr) {
    if (err) *err = "simple_pir DeserializeHint: hint_out is null";
    return retcode::FAIL;
  }
  size_t off = 0;
  if (blob.size() < 4) {
    if (err) *err = "simple_pir DeserializeHint: blob shorter than magic";
    return retcode::FAIL;
  }
  if (std::memcmp(blob.data(), kSimpleHintMagic, 4) != 0) {
    if (err) *err = "simple_pir DeserializeHint: bad magic (expected PSHB)";
    return retcode::FAIL;
  }
  off += 4;
  uint16_t version = 0, reserved = 0;
  if (!ReadU16(blob, &off, &version) || !ReadU16(blob, &off, &reserved)) {
    if (err) *err = "simple_pir DeserializeHint: truncated header";
    return retcode::FAIL;
  }
  if (version != kSimpleHintWireVersion) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: unsupported version " << version
          << " (this build understands " << kSimpleHintWireVersion << ")";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  if (reserved != 0) {
    if (err) *err = "simple_pir DeserializeHint: nonzero reserved field";
    return retcode::FAIL;
  }
  core::DBinfo info;
  if (!ReadU64(blob, &off, &info.num) ||
      !ReadU64(blob, &off, &info.row_length) ||
      !ReadU64(blob, &off, &info.packing) ||
      !ReadU64(blob, &off, &info.ne) ||
      !ReadU64(blob, &off, &info.x) ||
      !ReadU64(blob, &off, &info.p) ||
      !ReadU64(blob, &off, &info.logq) ||
      !ReadU64(blob, &off, &info.basis) ||
      !ReadU64(blob, &off, &info.squishing) ||
      !ReadU64(blob, &off, &info.cols)) {
    if (err) *err = "simple_pir DeserializeHint: truncated DBinfo";
    return retcode::FAIL;
  }
  hint_out->info_after_squish = info;

  auto rc = ReadMatrix(blob, &off, &hint_out->A, err, "A");
  if (rc != retcode::SUCCESS) return rc;
  rc = ReadMatrix(blob, &off, &hint_out->H, err, "H");
  if (rc != retcode::SUCCESS) return rc;

  if (off != blob.size()) {
    if (err) {
      std::ostringstream oss;
      oss << "simple_pir DeserializeHint: " << (blob.size() - off)
          << " trailing bytes after last matrix";
      *err = oss.str();
    }
    return retcode::FAIL;
  }
  return retcode::SUCCESS;
}

}  // namespace primihub::pir::simple_pir
