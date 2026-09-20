/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * primihub::pir::core::Matrix unit tests. Most tests are unconditional
 * — Get/Set/MatrixAdd etc. live outside the kernel switch. The three
 * kernel-bridge tests (Mul / MulVec / Transpose) bifurcate on
 * kPirCoreKernelsVendored: vendored mode asserts the math is correct
 * vs in-line expected; stub mode asserts the methods return FAIL with
 * the activation-flag hint.
 */
#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "src/primihub/kernel/pir/operator/pir_core/matrix.h"

#include "src/primihub/kernel/pir/operator/pir_core/lwe_params.h"

namespace primihub::pir::core {
namespace {

TEST(MatrixTest, ShapeAndAccessors) {
  Matrix m(3, 4);
  EXPECT_EQ(m.rows(), 3u);
  EXPECT_EQ(m.cols(), 4u);
  EXPECT_EQ(m.size(), 12u);
  EXPECT_FALSE(m.empty());
  // ctor zero-initializes.
  for (uint64_t i = 0; i < 3; ++i) {
    for (uint64_t j = 0; j < 4; ++j) {
      EXPECT_EQ(m.Get(i, j), 0u) << "i=" << i << " j=" << j;
    }
  }
  m.Set(1, 2, 42);
  EXPECT_EQ(m.Get(1, 2), 42u);
  // Other cells unchanged.
  EXPECT_EQ(m.Get(0, 0), 0u);
  EXPECT_EQ(m.Get(2, 3), 0u);
}

TEST(MatrixTest, ZerosFactory) {
  auto m = Matrix::Zeros(2, 5);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 5u);
  for (uint64_t i = 0; i < m.size(); ++i) {
    EXPECT_EQ(m.data()[i], 0u);
  }
}

TEST(MatrixTest, UniformRandomBound) {
  auto m = Matrix::UniformRandom(10, 10, /*logmod=*/8);
  EXPECT_EQ(m.size(), 100u);
  // logmod=8 caps each element to [0, 256). Cannot assert distribution,
  // but the bound is a hard invariant.
  for (uint64_t i = 0; i < m.size(); ++i) {
    EXPECT_LT(m.data()[i], 256u);
  }
}

TEST(MatrixTest, MatrixAddMatrixSub) {
  Matrix a(2, 2);
  a.Set(0, 0, 1); a.Set(0, 1, 2); a.Set(1, 0, 3); a.Set(1, 1, 4);
  Matrix b(2, 2);
  b.Set(0, 0, 10); b.Set(0, 1, 20); b.Set(1, 0, 30); b.Set(1, 1, 40);
  a.MatrixAdd(b);
  EXPECT_EQ(a.Get(0, 0), 11u);
  EXPECT_EQ(a.Get(0, 1), 22u);
  EXPECT_EQ(a.Get(1, 0), 33u);
  EXPECT_EQ(a.Get(1, 1), 44u);
  a.MatrixSub(b);
  EXPECT_EQ(a.Get(0, 0), 1u);
  EXPECT_EQ(a.Get(0, 1), 2u);
  EXPECT_EQ(a.Get(1, 0), 3u);
  EXPECT_EQ(a.Get(1, 1), 4u);
}

TEST(MatrixTest, ScalarArithmetic) {
  Matrix m(2, 2);
  m.Set(0, 0, 10); m.Set(0, 1, 20); m.Set(1, 0, 30); m.Set(1, 1, 40);
  m.ScalarAdd(5);
  EXPECT_EQ(m.Get(0, 0), 15u);
  EXPECT_EQ(m.Get(1, 1), 45u);
  m.ScalarSub(15);
  EXPECT_EQ(m.Get(0, 0), 0u);
  EXPECT_EQ(m.Get(1, 1), 30u);
}

TEST(MatrixTest, ReduceMod) {
  Matrix m(1, 3);
  m.Set(0, 0, 17); m.Set(0, 1, 256); m.Set(0, 2, 100);
  m.ReduceMod(10);
  EXPECT_EQ(m.Get(0, 0), 7u);
  EXPECT_EQ(m.Get(0, 1), 6u);
  EXPECT_EQ(m.Get(0, 2), 0u);
}

TEST(MatrixTest, DropLastRows) {
  Matrix m(5, 3);
  // Fill so dropped rows are observable in the size count.
  for (uint64_t i = 0; i < 5; ++i) {
    for (uint64_t j = 0; j < 3; ++j) {
      m.Set(i, j, static_cast<uint32_t>(i * 3 + j));
    }
  }
  m.DropLastRows(2);
  EXPECT_EQ(m.rows(), 3u);
  EXPECT_EQ(m.size(), 9u);
  // The first 3 rows are untouched.
  EXPECT_EQ(m.Get(0, 0), 0u);
  EXPECT_EQ(m.Get(2, 2), 8u);
}

TEST(MatrixTest, MulVecBifurcatesOnVendoring) {
  // 3x3 * 3x1 — 3 row sums.
  Matrix a(3, 3);
  for (uint64_t r = 0; r < 3; ++r) {
    for (uint64_t c = 0; c < 3; ++c) {
      a.Set(r, c, static_cast<uint32_t>(r * 3 + c));
    }
  }
  Matrix b(3, 1);
  b.Set(0, 0, 1); b.Set(1, 0, 1); b.Set(2, 0, 1);
  Matrix out;
  std::string err;
  auto rc = a.MulVec(b, &out, &err);
  if (kPirCoreKernelsVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS) << "vendored mode: " << err;
    EXPECT_EQ(out.rows(), 3u);
    EXPECT_EQ(out.cols(), 1u);
    // Expected row sums: [0+1+2, 3+4+5, 6+7+8] = [3, 12, 21].
    EXPECT_EQ(out.Get(0, 0), 3u);
    EXPECT_EQ(out.Get(1, 0), 12u);
    EXPECT_EQ(out.Get(2, 0), 21u);
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_pir_core_real=1"), std::string::npos)
        << "stub error must name the activation flag; got: " << err;
  }
}

TEST(MatrixTest, MulBifurcatesOnVendoring) {
  // 2x3 * 3x2 -> 2x2. Use a known small case.
  Matrix a(2, 3);
  // a = [[1,2,3], [4,5,6]]
  a.Set(0, 0, 1); a.Set(0, 1, 2); a.Set(0, 2, 3);
  a.Set(1, 0, 4); a.Set(1, 1, 5); a.Set(1, 2, 6);
  Matrix b(3, 2);
  // b = [[7,8], [9,10], [11,12]]
  b.Set(0, 0, 7);  b.Set(0, 1, 8);
  b.Set(1, 0, 9);  b.Set(1, 1, 10);
  b.Set(2, 0, 11); b.Set(2, 1, 12);
  Matrix out;
  std::string err;
  auto rc = a.Mul(b, &out, &err);
  if (kPirCoreKernelsVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS) << "vendored mode: " << err;
    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 2u);
    // 1*7+2*9+3*11 = 58; 1*8+2*10+3*12 = 64;
    // 4*7+5*9+6*11 = 139; 4*8+5*10+6*12 = 154.
    EXPECT_EQ(out.Get(0, 0), 58u);
    EXPECT_EQ(out.Get(0, 1), 64u);
    EXPECT_EQ(out.Get(1, 0), 139u);
    EXPECT_EQ(out.Get(1, 1), 154u);
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_pir_core_real=1"), std::string::npos);
  }
}

TEST(MatrixTest, TransposeBifurcatesOnVendoring) {
  Matrix a(2, 3);
  a.Set(0, 0, 1); a.Set(0, 1, 2); a.Set(0, 2, 3);
  a.Set(1, 0, 4); a.Set(1, 1, 5); a.Set(1, 2, 6);
  Matrix out;
  std::string err;
  auto rc = a.Transpose(&out, &err);
  if (kPirCoreKernelsVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS) << "vendored mode: " << err;
    EXPECT_EQ(out.rows(), 3u);
    EXPECT_EQ(out.cols(), 2u);
    EXPECT_EQ(out.Get(0, 0), 1u);
    EXPECT_EQ(out.Get(0, 1), 4u);
    EXPECT_EQ(out.Get(1, 0), 2u);
    EXPECT_EQ(out.Get(1, 1), 5u);
    EXPECT_EQ(out.Get(2, 0), 3u);
    EXPECT_EQ(out.Get(2, 1), 6u);
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("enable_pir_core_real=1"), std::string::npos);
  }
}

TEST(MatrixTest, MulVecDimMismatchFailsWithoutKernelCall) {
  // Even in vendored mode, dimension mismatches must FAIL up-front
  // before touching the kernel.
  Matrix a(3, 3);
  Matrix b(2, 1);  // wrong row count
  Matrix out;
  std::string err;
  auto rc = a.MulVec(b, &out, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("dim mismatch"), std::string::npos)
      << "expected dim mismatch error; got: " << err;
}

TEST(MatrixTest, MulVecNonColumnFailsUpFront) {
  Matrix a(3, 3);
  Matrix b(3, 2);  // not a column vector
  Matrix out;
  std::string err;
  auto rc = a.MulVec(b, &out, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("column vector"), std::string::npos)
      << "expected column-vector error; got: " << err;
}

// --------------------------------------------------------------------
// Squish / Unsquish — pure arithmetic, both modes equivalent.
// --------------------------------------------------------------------

TEST(MatrixTest, SquishPacksWithinSlots) {
  // 2 x 6, basis=4, delta=3 — packs 3 cols per output cell; output is
  // 2 x 2. Each cell low 12 bits = old[i, 3j+0] | old[i, 3j+1]<<4 |
  // old[i, 3j+2]<<8.
  Matrix m(2, 6);
  for (uint64_t i = 0; i < 2; ++i) {
    for (uint64_t j = 0; j < 6; ++j) {
      m.Set(i, j, static_cast<uint32_t>(i * 6 + j + 1));
    }
  }
  m.Squish(/*basis=*/4, /*delta=*/3);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 2u);
  // Row 0 col 0: 1 | 2<<4 | 3<<8 = 0x321
  EXPECT_EQ(m.Get(0, 0), 0x321u);
  // Row 0 col 1: 4 | 5<<4 | 6<<8 = 0x654
  EXPECT_EQ(m.Get(0, 1), 0x654u);
  // Row 1 col 0: 7 | 8<<4 | 9<<8 = 0x987
  EXPECT_EQ(m.Get(1, 0), 0x987u);
  // Row 1 col 1: A | B<<4 | C<<8 = 0xCBA
  EXPECT_EQ(m.Get(1, 1), 0xCBAu);
}

TEST(MatrixTest, SquishHandlesNonDivisibleCols) {
  // 1 x 5, delta=3 — output 1 x 2; second cell has only 2 inputs.
  Matrix m(1, 5);
  for (uint64_t j = 0; j < 5; ++j) m.Set(0, j, static_cast<uint32_t>(j + 1));
  m.Squish(/*basis=*/4, /*delta=*/3);
  EXPECT_EQ(m.rows(), 1u);
  EXPECT_EQ(m.cols(), 2u);
  EXPECT_EQ(m.Get(0, 0), 0x321u);
  // Second cell: only old cols 3, 4 contribute (delta*1+0=3, +1=4; +2=5 OOB).
  // Value: 4 | 5<<4 | 0<<8 = 0x54.
  EXPECT_EQ(m.Get(0, 1), 0x54u);
}

TEST(MatrixTest, SquishThenUnsquishRoundtrip) {
  // 3 x 7, basis=10, delta=3 — upstream's chosen params. Values must
  // fit in `basis` bits (so < 1024) for the roundtrip to be exact.
  Matrix m(3, 7);
  for (uint64_t i = 0; i < 3; ++i) {
    for (uint64_t j = 0; j < 7; ++j) {
      m.Set(i, j, static_cast<uint32_t>((i * 100 + j) % 1024));
    }
  }
  Matrix original = m;  // value copy of internal vector

  m.Squish(/*basis=*/10, /*delta=*/3);
  // ceil(7/3) = 3 packed cols.
  EXPECT_EQ(m.cols(), 3u);

  m.Unsquish(/*basis=*/10, /*delta=*/3, /*orig_cols=*/7);
  ASSERT_EQ(m.rows(), original.rows());
  ASSERT_EQ(m.cols(), original.cols());
  for (uint64_t i = 0; i < m.rows(); ++i) {
    for (uint64_t j = 0; j < m.cols(); ++j) {
      EXPECT_EQ(m.Get(i, j), original.Get(i, j))
          << "i=" << i << " j=" << j;
    }
  }
}


// ---- AppendZeros (chunk 1 of DoublePIR port; task 5.5 dep) ----

TEST(MatrixTest, AppendZerosAddsZeroRowsToColumnVector) {
  Matrix v(3, 1);
  v.Set(0, 0, 7);
  v.Set(1, 0, 8);
  v.Set(2, 0, 9);
  v.AppendZeros(2);
  EXPECT_EQ(v.rows(), 5u);
  EXPECT_EQ(v.cols(), 1u);
  EXPECT_EQ(v.Get(0, 0), 7u);
  EXPECT_EQ(v.Get(1, 0), 8u);
  EXPECT_EQ(v.Get(2, 0), 9u);
  EXPECT_EQ(v.Get(3, 0), 0u);
  EXPECT_EQ(v.Get(4, 0), 0u);
}

TEST(MatrixTest, AppendZerosZeroIsNoOp) {
  Matrix v(2, 1);
  v.Set(0, 0, 11);
  v.Set(1, 0, 22);
  v.AppendZeros(0);
  EXPECT_EQ(v.rows(), 2u);
  EXPECT_EQ(v.Get(0, 0), 11u);
  EXPECT_EQ(v.Get(1, 0), 22u);
}

TEST(MatrixDeathTest, AppendZerosFatalOnMultiColumn) {
  Matrix m(2, 3);
  EXPECT_DEATH(m.AppendZeros(1), "AppendZeros");
}

// ---- ConcatCols ----

TEST(MatrixTest, ConcatColsOneIsNoOp) {
  Matrix m(2, 3);
  m.Set(0, 0, 1); m.Set(0, 1, 2); m.Set(0, 2, 3);
  m.Set(1, 0, 4); m.Set(1, 1, 5); m.Set(1, 2, 6);
  m.ConcatCols(1);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 3u);
  EXPECT_EQ(m.Get(0, 1), 2u);
  EXPECT_EQ(m.Get(1, 2), 6u);
}

TEST(MatrixTest, ConcatColsFoldsByNVerticallyStackingColumnGroups) {
  // Input 2x6, n=3 -> output (2*3)x(6/3) = 6x2.
  // Upstream rule: new[i + rows*(j%n), j/n] = old[i, j].
  // Input layout:
  //   row 0: 1  2  3  4  5  6
  //   row 1: 7  8  9 10 11 12
  // Output column 0 (from input cols j=0,1,2):
  //   row 0..1: input col 0 -> [1, 7]
  //   row 2..3: input col 1 -> [2, 8]
  //   row 4..5: input col 2 -> [3, 9]
  // Output column 1 (from input cols j=3,4,5):
  //   row 0..1: input col 3 -> [4, 10]
  //   row 2..3: input col 4 -> [5, 11]
  //   row 4..5: input col 5 -> [6, 12]
  Matrix m(2, 6);
  uint32_t v = 1;
  for (uint64_t i = 0; i < 2; ++i) {
    for (uint64_t j = 0; j < 6; ++j) {
      m.Set(i, j, v++);
    }
  }
  m.ConcatCols(3);
  EXPECT_EQ(m.rows(), 6u);
  EXPECT_EQ(m.cols(), 2u);
  // Column 0
  EXPECT_EQ(m.Get(0, 0), 1u);
  EXPECT_EQ(m.Get(1, 0), 7u);
  EXPECT_EQ(m.Get(2, 0), 2u);
  EXPECT_EQ(m.Get(3, 0), 8u);
  EXPECT_EQ(m.Get(4, 0), 3u);
  EXPECT_EQ(m.Get(5, 0), 9u);
  // Column 1
  EXPECT_EQ(m.Get(0, 1), 4u);
  EXPECT_EQ(m.Get(1, 1), 10u);
  EXPECT_EQ(m.Get(2, 1), 5u);
  EXPECT_EQ(m.Get(3, 1), 11u);
  EXPECT_EQ(m.Get(4, 1), 6u);
  EXPECT_EQ(m.Get(5, 1), 12u);
}

TEST(MatrixDeathTest, ConcatColsFatalOnNonDivisor) {
  Matrix m(2, 5);
  EXPECT_DEATH(m.ConcatCols(2), "ConcatCols");
}

TEST(MatrixDeathTest, ConcatColsZeroFatal) {
  Matrix m(2, 2);
  EXPECT_DEATH(m.ConcatCols(0), "ConcatCols");
}


// ---- Expand / Contract (chunk 2 of DoublePIR port; task 5.5 dep) ----
//
// These tests pin each method to upstream simplepir's matrix.go
// verbatim. Round-trip (Contract(Expand(x)) == x) is intentionally NOT
// tested — see the Contract docstring: the uint32 underflow Expand
// uses to encode negative digits introduces a per-digit offset of
// (2^32 mod p) when Contract reads it back, so the round-trip is
// exact only for p that divides 2^32. DoublePIR uses arbitrary p
// (e.g. 929, 781) and handles this by applying Contract within an
// LWE protocol context where the offset cancels.

TEST(MatrixTest, ExpandWritesBasePDigitsCenteredByPOver2) {
  // p = 4, delta = 2. p/2 = 2. Each input cell becomes 2 rows.
  // Input 1x2:
  //   row 0: 7 5
  // 7 in base 4 = (3, 1)  ->  centered: (3-2, 1-2) = (1, -1) =
  //                                                    (1, 0xFFFFFFFF)
  // 5 in base 4 = (1, 1)  ->  centered: (1-2, 1-2) = (-1, -1) =
  //                                                    (0xFFFFFFFF, 0xFFFFFFFF)
  Matrix m(1, 2);
  m.Set(0, 0, 7);
  m.Set(0, 1, 5);
  m.Expand(4, 2);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 2u);
  EXPECT_EQ(m.Get(0, 0), 1u);
  EXPECT_EQ(m.Get(1, 0), 0xFFFFFFFFu);
  EXPECT_EQ(m.Get(0, 1), 0xFFFFFFFFu);
  EXPECT_EQ(m.Get(1, 1), 0xFFFFFFFFu);
}

TEST(MatrixTest, ExpandGrowsRowsByDeltaWithColsUnchanged) {
  Matrix m(5, 3);
  m.Expand(7, 4);
  EXPECT_EQ(m.rows(), 20u);
  EXPECT_EQ(m.cols(), 3u);
}

TEST(MatrixTest, ContractMatchesUpstreamReconstructFromBaseP) {
  // Hand-build the expanded form (the per-digit centered layout
  // Expand would have produced) and verify Contract reconstructs the
  // expected values. p = 5, delta = 3, p/2 = 2.
  // Target output values: 17 and 11.
  //   17 = 2 + 3*5 + 0*25 -> digits (2, 3, 0) -> centered (0, 1, -2) =
  //                              (0, 1, 0xFFFFFFFE)
  //   11 = 1 + 2*5 + 0*25 -> digits (1, 2, 0) -> centered (-1, 0, -2) =
  //                              (0xFFFFFFFF, 0, 0xFFFFFFFE)
  // Contract reads each digit back: (uint64(stored) + 2) % 5
  //   col 0:
  //     row 0 stored 0          -> (0+2)%5 = 2
  //     row 1 stored 1          -> (1+2)%5 = 3
  //     row 2 stored 0xFFFFFFFE -> (0xFFFFFFFE+2)%5 = 0x100000000 % 5
  //   col 1:
  //     row 0 stored 0xFFFFFFFF -> (0xFFFFFFFF+2)%5 = 0x100000001 % 5
  //     row 1 stored 0          -> (0+2)%5 = 2
  //     row 2 stored 0xFFFFFFFE -> 0x100000000 % 5
  // 0x100000000 = 4294967296. 4294967296 % 5 = 1 (since 2^32 mod 5 = 1).
  // 0x100000001 % 5 = 2.
  // col 0: vals = [2, 3, 1]; sum = 2 + 3*5 + 1*25 = 2 + 15 + 25 = 42.
  // col 1: vals = [2, 2, 1]; sum = 2 + 2*5 + 1*25 = 2 + 10 + 25 = 37.
  // (Round-trip "value 17" -> 42 illustrates the offset effect — the
  // test asserts the algorithm, not the round-trip.)
  Matrix m(3, 2);
  m.Set(0, 0, 0);             m.Set(0, 1, 0xFFFFFFFFu);
  m.Set(1, 0, 1);             m.Set(1, 1, 0);
  m.Set(2, 0, 0xFFFFFFFEu);   m.Set(2, 1, 0xFFFFFFFEu);
  m.Contract(5, 3);
  EXPECT_EQ(m.rows(), 1u);
  EXPECT_EQ(m.cols(), 2u);
  EXPECT_EQ(m.Get(0, 0), 42u);
  EXPECT_EQ(m.Get(0, 1), 37u);
}

TEST(MatrixTest, ContractRoundTripsExactlyWhenPIsPowerOfTwo) {
  // The one regime where Expand+Contract IS a clean inverse — when p
  // divides 2^32 evenly, the per-digit wrap-around offset is 0.
  // p = 4 (2^32 mod 4 == 0), delta = 3, p^delta = 64. Values 0..15
  // are well within bounds and round-trip exactly.
  Matrix m(2, 4);
  uint32_t v = 0;
  for (uint64_t i = 0; i < 2; ++i) {
    for (uint64_t j = 0; j < 4; ++j) {
      m.Set(i, j, v++);
    }
  }
  Matrix original(2, 4);
  for (uint64_t i = 0; i < 2; ++i) {
    for (uint64_t j = 0; j < 4; ++j) {
      original.Set(i, j, m.Get(i, j));
    }
  }
  m.Expand(4, 3);
  EXPECT_EQ(m.rows(), 6u);
  EXPECT_EQ(m.cols(), 4u);
  m.Contract(4, 3);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 4u);
  for (uint64_t i = 0; i < 2; ++i) {
    for (uint64_t j = 0; j < 4; ++j) {
      EXPECT_EQ(m.Get(i, j), original.Get(i, j))
          << "p=4 round-trip mismatch at (" << i << ", " << j << ")";
    }
  }
}

TEST(MatrixDeathTest, ExpandFatalOnZeroP) {
  Matrix m(2, 2);
  EXPECT_DEATH(m.Expand(0, 2), "Expand");
}

TEST(MatrixDeathTest, ExpandFatalOnPOne) {
  Matrix m(2, 2);
  EXPECT_DEATH(m.Expand(1, 2), "Expand");
}

TEST(MatrixDeathTest, ExpandFatalOnZeroDelta) {
  Matrix m(2, 2);
  EXPECT_DEATH(m.Expand(4, 0), "Expand");
}

TEST(MatrixDeathTest, ContractFatalOnRowsNotDivisibleByDelta) {
  Matrix m(5, 2);
  EXPECT_DEATH(m.Contract(4, 2), "Contract");
}


// ---- Concat (chunk 4a of DoublePIR port; task 5.5 dep) ----

TEST(MatrixTest, ConcatAppendsRowsWhenColsMatch) {
  Matrix top(2, 3);
  top.Set(0, 0, 1); top.Set(0, 1, 2); top.Set(0, 2, 3);
  top.Set(1, 0, 4); top.Set(1, 1, 5); top.Set(1, 2, 6);
  Matrix bot(1, 3);
  bot.Set(0, 0, 7); bot.Set(0, 1, 8); bot.Set(0, 2, 9);
  top.Concat(bot);
  EXPECT_EQ(top.rows(), 3u);
  EXPECT_EQ(top.cols(), 3u);
  EXPECT_EQ(top.Get(0, 0), 1u); EXPECT_EQ(top.Get(0, 2), 3u);
  EXPECT_EQ(top.Get(1, 0), 4u); EXPECT_EQ(top.Get(1, 2), 6u);
  EXPECT_EQ(top.Get(2, 0), 7u); EXPECT_EQ(top.Get(2, 2), 9u);
}

TEST(MatrixTest, ConcatAdoptsShapeWhenEmpty) {
  Matrix empty;
  Matrix src(2, 3);
  src.Set(0, 0, 11); src.Set(1, 2, 22);
  empty.Concat(src);
  EXPECT_EQ(empty.rows(), 2u);
  EXPECT_EQ(empty.cols(), 3u);
  EXPECT_EQ(empty.Get(0, 0), 11u);
  EXPECT_EQ(empty.Get(1, 2), 22u);
}

TEST(MatrixTest, ConcatEmptyAppendIsNoOp) {
  Matrix m(2, 3);
  m.Set(0, 0, 5);
  Matrix empty_rows(0, 3);  // 0 rows, cols irrelevant for no-op path
  m.Concat(empty_rows);
  EXPECT_EQ(m.rows(), 2u);
  EXPECT_EQ(m.cols(), 3u);
  EXPECT_EQ(m.Get(0, 0), 5u);
}

TEST(MatrixDeathTest, ConcatFatalOnColMismatch) {
  Matrix a(2, 3);
  Matrix b(1, 4);
  EXPECT_DEATH(a.Concat(b), "Concat cols mismatch");
}


// ---- SelectRows / Round / MulTransposedPacked (chunk 6a) ----

TEST(MatrixTest, SelectRowsReturnsDeepCopyOfSlice) {
  Matrix m(4, 3);
  uint32_t v = 1;
  for (uint64_t i = 0; i < 4; ++i) {
    for (uint64_t j = 0; j < 3; ++j) {
      m.Set(i, j, v++);
    }
  }
  Matrix slice = m.SelectRows(1, 2);
  EXPECT_EQ(slice.rows(), 2u);
  EXPECT_EQ(slice.cols(), 3u);
  EXPECT_EQ(slice.Get(0, 0), 4u);  // original row 1
  EXPECT_EQ(slice.Get(0, 2), 6u);
  EXPECT_EQ(slice.Get(1, 0), 7u);  // original row 2
  EXPECT_EQ(slice.Get(1, 2), 9u);
  // Deep copy: mutating the slice must not touch the original.
  slice.Set(0, 0, 9999u);
  EXPECT_EQ(m.Get(1, 0), 4u);
}

TEST(MatrixTest, SelectRowsZeroCountReturnsEmpty) {
  Matrix m(3, 2);
  Matrix slice = m.SelectRows(1, 0);
  EXPECT_EQ(slice.rows(), 0u);
  EXPECT_EQ(slice.cols(), 2u);
}

TEST(MatrixDeathTest, SelectRowsFatalOnOutOfBounds) {
  Matrix m(3, 2);
  EXPECT_DEATH(m.SelectRows(2, 2), "SelectRows");
}

TEST(MatrixTest, RoundAppliesLweParamsRoundToEachCell) {
  // Pick params with a tight Round behaviour: p = 4, logq = 32 makes
  // Delta = 2^30. LweParams::Round(x) = ((x + Delta/2) / Delta) % p.
  // Sample input cell values that map to known Round outputs.
  LweParams params;
  params.logq = 32;
  params.p = 4;
  const uint32_t delta = static_cast<uint32_t>(params.Delta());
  Matrix m(2, 2);
  m.Set(0, 0, 0u);                 // Round(0) = 0
  m.Set(0, 1, delta);              // Round(Delta) = 1
  m.Set(1, 0, 2u * delta);         // Round(2*Delta) = 2
  m.Set(1, 1, 3u * delta);         // Round(3*Delta) = 3
  m.Round(params);
  EXPECT_EQ(m.Get(0, 0), 0u);
  EXPECT_EQ(m.Get(0, 1), 1u);
  EXPECT_EQ(m.Get(1, 0), 2u);
  EXPECT_EQ(m.Get(1, 1), 3u);
}

TEST(MatrixTest, MulTransposedPackedRejectsNonHardcodedParams) {
  // Stub-or-vendored: input-validation happens before the kernel
  // dispatch, so the EXPECT_EQ holds in both modes.
  Matrix a(4, 2);
  Matrix b(4, 2);
  Matrix out;
  std::string err;
  EXPECT_EQ(a.MulTransposedPacked(b, 8, 3, &out, &err), retcode::FAIL);
  EXPECT_NE(err.find("basis=8"), std::string::npos) << err;
}

TEST(MatrixTest, MulTransposedPackedRejectsBadColLayout) {
  // a is packed 4x2 → b must be unpacked 4x6 (cols = 2*3). 4x3
  // (one packed col instead of three) trips the up-front guard.
  Matrix a(4, 2);
  Matrix b(8, 3);
  Matrix out;
  std::string err;
  EXPECT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::FAIL);
  EXPECT_NE(err.find("b.cols must equal"), std::string::npos) << err;
}

TEST(MatrixTest, MulTransposedPackedRejectsBadBRowAlignmentInShortRowsBranch) {
  // bRows must be a multiple of 8 ONLY when aRows <= aCols (kernel
  // short-rows branch j+=8 unroll). aRows=2, aCols=4 → short-rows;
  // bRows=7 trips the guard.
  Matrix a(2, 4);
  Matrix b(7, 12);
  Matrix out;
  std::string err;
  EXPECT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::FAIL);
  EXPECT_NE(err.find("multiple of 8"), std::string::npos) << err;
}

TEST(MatrixTest, MulTransposedPackedAllowsAnyBRowsInLongRowsBranch) {
  // Long-rows branch (aRows > aCols) steps j by 1 — bRows=4 must be
  // accepted. We're testing input validation only, so stub mode is
  // fine.
  Matrix a(4, 2);
  Matrix b(4, 6);
  Matrix out;
  std::string err;
  if (kPirCoreKernelsVendored) {
    EXPECT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::SUCCESS)
        << err;
  } else {
    EXPECT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::FAIL);
    EXPECT_NE(err.find("not vendored"), std::string::npos) << err;
  }
}

TEST(MatrixTest, MulTransposedPackedFailsLoudlyInStubMode) {
  if (kPirCoreKernelsVendored) {
    GTEST_SKIP() << "vendored mode succeeds — see MulTransposedPackedSmokeProducesCorrectShape";
  }
  Matrix a(4, 2);
  Matrix b(8, 6);  // valid layout: bCols = aCols*3, bRows % 8 == 0
  Matrix out;
  std::string err;
  EXPECT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::FAIL);
  EXPECT_NE(err.find("not vendored"), std::string::npos) << err;
}

TEST(MatrixTest, MulTransposedPackedSmokeProducesCorrectShape) {
  if (!kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs kernel bridge";
  }
  // Realistic DoublePIR shape: a is squished (aRows x aCols), b is
  // unpacked (bRows x bCols) with bCols == aCols * 3. aRows > aCols
  // exercises the "long rows" kernel branch. Output is aRows x bRows.
  Matrix a(8, 2);
  Matrix b(8, 6);
  a.Set(0, 0, 1u);
  b.Set(0, 0, 1u);
  Matrix out;
  std::string err;
  ASSERT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::SUCCESS)
      << err;
  EXPECT_EQ(out.rows(), 8u);
  EXPECT_EQ(out.cols(), 8u);
}

TEST(MatrixTest, MulTransposedPackedShortRowsBranchProducesCorrectShape) {
  if (!kPirCoreKernelsVendored) {
    GTEST_SKIP() << "needs kernel bridge";
  }
  // aRows <= aCols exercises the "short rows" kernel branch which
  // unrolls the j-loop by 8 — so bRows must be a multiple of 8.
  Matrix a(2, 4);
  Matrix b(8, 12);
  a.Set(0, 0, 1u);
  b.Set(0, 0, 1u);
  Matrix out;
  std::string err;
  ASSERT_EQ(a.MulTransposedPacked(b, 10, 3, &out, &err), retcode::SUCCESS)
      << err;
  EXPECT_EQ(out.rows(), 2u);
  EXPECT_EQ(out.cols(), 8u);
}

}  // namespace
}  // namespace primihub::pir::core
