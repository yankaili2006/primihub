/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_kernel.h"

#include <cstdint>

namespace primihub::pir::ypir {

std::uint64_t CrtCompose2(const Params& p, std::uint64_t x, std::uint64_t y) {
  __uint128_t val = static_cast<__uint128_t>(x) * p.mod1_inv_mod0;
  val += static_cast<__uint128_t>(y) * p.mod0_inv_mod1;
  return BarrettReductionU128Raw(p.modulus, p.barrett_cr_0_modulus,
                                 p.barrett_cr_1_modulus, val);
}

}  // namespace primihub::pir::ypir
