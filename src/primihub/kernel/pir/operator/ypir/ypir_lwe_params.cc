/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 */
#include "src/primihub/kernel/pir/operator/ypir/ypir_lwe_params.h"

namespace primihub::pir::ypir {

LweParams LweParams::Default() {
  LweParams p;
  p.n = 1024;
  // `1u64 << 32` in Rust. Use the explicit constant rather than
  // `1ull << 32` to keep the intent (q = 2^32) close to upstream.
  p.modulus = static_cast<std::uint64_t>(1) << 32;
  p.pt_modulus = 256;
  p.q2_bits = 28;
  p.noise_width = kDefaultLweNoiseWidth;
  return p;
}

}  // namespace primihub::pir::ypir
