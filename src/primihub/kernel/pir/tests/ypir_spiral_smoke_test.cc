/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * ypir_spiral_smoke_test — proves a ypir translation unit can link the
 * C++ Spiral library (@spiral_pir//:spiral_lib, now buildable since
 * Intel HEXL was wired into the WORKSPACE). This is the foundation for
 * the Spiral-coupled YPIR chunks (kernel/convolution/params/packing/...
 * which need MatPoly / NTT / the compile-time Params). It exercises the
 * minimal API surface — the compile-time `poly_len` global, a 2x1 raw
 * MatPoly round-trip, and `rescale` — and cross-checks that Spiral's
 * `rescale` equals our standalone port (ypir_modulus_switch Rescale).
 *
 * Built behind tags=["manual"] so wildcard runs don't force the heavy
 * Spiral/HEXL compile; run explicitly:
 *   bazel test --config=linux_x86_64 \
 *     //src/primihub/kernel/pir/tests:ypir_spiral_smoke_test
 */
#include <cstdint>

#include <gtest/gtest.h>

#include "spiral.h"  // @spiral_pir, brings MatPoly / rescale / poly_len

namespace {

TEST(YpirSpiralSmokeTest, LinksSpiralAndBasicMatPolyOps) {
  // poly_len is a compile-time constant from values.h, fixed by the
  // SPIRAL_DEFINES "wiki" config propagated by spiral_lib.
  EXPECT_EQ(poly_len, 2048u);

  // Construct a 2x1 raw (non-NTT) polynomial matrix and round-trip
  // coefficients through the operator[] row accessor.
  MatPoly m(2, 1, false);
  m[0][0] = 12345u;
  m[1][0] = 67890u;
  EXPECT_EQ(m[0][0], 12345u);
  EXPECT_EQ(m[1][0], 67890u);

  // Spiral's rescale (poly.cpp) — cross-check against the values our
  // standalone Rescale port (ypir_modulus_switch) was verified against.
  EXPECT_EQ(rescale(8, 64, 8, false), 1u);
  EXPECT_EQ(rescale(40, 64, 8, false), 5u);
}

}  // namespace
