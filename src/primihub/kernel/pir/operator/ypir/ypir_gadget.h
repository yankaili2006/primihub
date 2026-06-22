/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_gadget — P4 of the spiral_rs Params/PolyMatrix port. Ports the
 * Galois automorphism (spiral-rs poly.rs automorph) and the gadget
 * matrix / gadget decomposition (spiral-rs gadget.rs). All pure
 * arithmetic on PolyMatrixRaw (coefficient form) — no HEXL.
 */
#ifndef SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_GADGET_H_
#define SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_GADGET_H_

#include <cstddef>

#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly_types.h"

namespace primihub::pir::ypir {

// Galois automorphism X -> X^t on each polynomial (negacyclic: a sign
// flip mod `modulus` when the wrapped index passes an odd multiple of
// poly_len). Mirrors automorph()/automorph_poly. Same shape as `a`.
PolyMatrixRaw Automorph(const Params& p, const PolyMatrixRaw& a,
                        std::size_t t);

// Bits per gadget element for a decomposition of `dim` digits.
std::size_t GetBitsPer(const Params& p, std::size_t dim);

// Power-of-(2^bits_per) gadget matrix, rows x cols (cols % rows == 0).
PolyMatrixRaw BuildGadget(const Params& p, std::size_t rows, std::size_t cols);

// Gadget decomposition with an explicit `rdim` (mirrors gadget_invert_rdim):
// only the first `rdim` rows of `inp` are decomposed, each coefficient into
// mx/rdim base-2^bits_per digits, written to out rows j + k*rdim. With
// rdim=1 this decomposes only inp row 0 (as the GSW key-switch needs).
PolyMatrixRaw GadgetInvertRdim(const Params& p, std::size_t mx,
                               const PolyMatrixRaw& inp, std::size_t rdim);

// Gadget decomposition of `inp` into `mx` rows (mx % inp.rows == 0):
// GadgetInvertRdim with rdim = inp.rows. Mirrors gadget_invert_alloc.
PolyMatrixRaw GadgetInvert(const Params& p, std::size_t mx,
                           const PolyMatrixRaw& inp);

}  // namespace primihub::pir::ypir

#endif  // SRC_PRIMIHUB_KERNEL_PIR_OPERATOR_YPIR_YPIR_GADGET_H_
