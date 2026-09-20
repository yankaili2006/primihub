/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_poly — P2 of the spiral_rs Params/PolyMatrix port (see
 * docs/pir/params-port-plan.md). Ports spiral-rs PolyMatrixRaw /
 * PolyMatrixNTT containers and the ntt()/raw() transforms.
 *
 * The NTT itself is offloaded to HEXL (NttContext holds one
 * intel::hexl::NTT per CRT modulus, built from Params). As in chunk 6,
 * the NTT-domain ordering is HEXL's own — fine as long as every
 * NTT-domain op goes through this module (self-consistent). The raw
 * (coefficient) form matches spiral_rs: from_ntt does per-modulus
 * inverse NTT + CRT reconstruction via the Params idempotents
 * (mod0_inv_mod1 / mod1_inv_mod0) and 128-bit Barrett, yielding each
 * coefficient in [0, modulus).
 *
 * Data layout (mirrors spiral_rs):
 *   raw  data: rows*cols*poly_len u64; poly(r,c) at (r*cols+c)*poly_len.
 *   ntt  data: rows*cols*crt_count*poly_len; poly(r,c) modulus m at
 *              (r*cols+c)*crt_count*poly_len + m*poly_len.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "hexl/ntt/ntt.hpp"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Holds the HEXL NTT contexts for a given Params (one per CRT modulus).
class NttContext {
 public:
  explicit NttContext(const Params& params);

  const Params& params() const { return params_; }

  PolyMatrixRaw ZeroRaw(std::size_t rows, std::size_t cols) const;
  PolyMatrixNTT ZeroNtt(std::size_t rows, std::size_t cols) const;

  // Forward transform: coefficient form -> NTT form (per modulus).
  PolyMatrixNTT ToNtt(const PolyMatrixRaw& in) const;
  // Inverse transform + CRT reconstruction: NTT form -> coefficient form
  // (each coefficient in [0, modulus)).
  PolyMatrixRaw FromNtt(const PolyMatrixNTT& in) const;

 private:
  Params params_;
  std::unique_ptr<intel::hexl::NTT> ntt_[kMaxModuli];
};

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_POLY_H_
