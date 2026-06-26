/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_server — see ypir_server.h for scope notes (sub-chunk 10a).
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"

#include <cassert>
#include <vector>

#include "src/primihub/kernel/pir/operator/ypir/ypir_bits.h"

namespace primihub::pir::ypir {

YConstants GenerateYConstants(const NttContext& ctx) {
  const Params& params = ctx.params();
  YConstants out;
  // for num_cts_log2 in 1..=poly_len_log2
  for (std::size_t num_cts_log2 = 1; num_cts_log2 <= params.poly_len_log2;
       ++num_cts_log2) {
    const std::size_t num_cts = static_cast<std::size_t>(1) << num_cts_log2;
    const std::size_t exp = params.poly_len / num_cts;  // Y = X^(poly_len/num_cts)

    PolyMatrixRaw y_raw = ctx.ZeroRaw(1, 1);
    y_raw.Poly(0, 0, params.poly_len)[exp] = 1;
    out.y.push_back(ctx.ToNtt(y_raw));

    PolyMatrixRaw neg_y_raw = ctx.ZeroRaw(1, 1);
    neg_y_raw.Poly(0, 0, params.poly_len)[exp] = params.modulus - 1;
    out.neg_y.push_back(ctx.ToNtt(neg_y_raw));
  }
  return out;
}

std::vector<std::uint16_t> SplitAlloc(const std::vector<std::uint64_t>& buf,
                                      std::size_t special_bit_offs,
                                      std::size_t rows, std::size_t cols,
                                      std::size_t out_rows,
                                      std::size_t inp_mod_bits,
                                      std::size_t pt_bits) {
  std::vector<std::uint16_t> out(out_rows * cols, 0);

  assert(out_rows >= rows);
  assert(inp_mod_bits >= pt_bits);

  const std::size_t scratch_len = out_rows * inp_mod_bits / 8;
  const std::uint64_t pt_mask =
      (static_cast<std::uint64_t>(1) << pt_bits) - 1;

  for (std::size_t j = 0; j < cols; ++j) {
    std::vector<std::uint8_t> bytes_tmp(scratch_len, 0);

    // read this column into the scratch bitstream
    std::size_t bit_offs = 0;
    for (std::size_t i = 0; i < rows; ++i) {
      const std::uint64_t inp = buf[i * cols + j];
      if (i == rows - 1) {
        bit_offs = special_bit_offs;
      }
      WriteBits(bytes_tmp.data(), bytes_tmp.size(), inp, bit_offs, inp_mod_bits);
      bit_offs += inp_mod_bits;
    }

    // now, 'stretch' the column vertically (re-chunk at pt_bits)
    bit_offs = 0;
    for (std::size_t i = 0; i < out_rows; ++i) {
      const std::uint64_t out_val =
          ReadBits(bytes_tmp.data(), bytes_tmp.size(), bit_offs, pt_bits);
      out[i * cols + j] = static_cast<std::uint16_t>(out_val);
      bit_offs += pt_bits;
      if (bit_offs >= out_rows * inp_mod_bits) {
        break;
      }
    }

    // upstream invariant: the value re-read at the special offset equals
    // the low pt_bits of the last input row.
    assert(static_cast<std::uint64_t>(out[(special_bit_offs / pt_bits) * cols + j]) ==
           (buf[(rows - 1) * cols + j] & pt_mask));
    (void)pt_mask;
  }

  return out;
}

std::size_t DbRowsPadded(const Params& params, bool pad_rows) {
  // The padded formula (db_rows + db_rows/(16*8)) is commented out
  // upstream, so both branches currently yield db_rows.
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (params.db_dim_1 + params.poly_len_log2);
  (void)pad_rows;
  return db_rows;
}

std::size_t DbCols(const Params& params, bool is_simplepir) {
  if (is_simplepir) {
    return params.instances * params.poly_len;
  }
  return static_cast<std::size_t>(1) << (params.db_dim_2 + params.poly_len_log2);
}

}  // namespace primihub::pir::ypir
