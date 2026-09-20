/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * End-to-end LHE-on-SimplePIR retrieval test (tiptoe chunk 1.1f). Plants a DB,
 * retrieves every entry through the full BFV-on-SimplePIR + LHE-hint path, and
 * checks exact recovery. Real mode needs SEAL (GFW-blocked on .50): manual +
 * gated on --define=enable_tiptoe_real=1; without it compiles to a skipped
 * test. Validated standalone vs SEAL 4.1 when this chunk landed (6x6 DB,
 * 0/36 mismatches).
 */
#include "gtest/gtest.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED

#include <cstdint>
#include <vector>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_lhe_pir.h"

namespace primihub::pir::tiptoe {
namespace {

TEST(TiptoeLhePirTest, RetrievesEveryEntryExactly) {
  const std::uint64_t m = 6, n = 512;
  std::vector<std::uint8_t> db(m * m);
  for (std::size_t i = 0; i < db.size(); ++i)
    db[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0xFF);

  const LheSimplePir pir(db, m, n, /*seed=*/123456789ull);

  for (std::uint64_t r = 0; r < m; ++r)
    for (std::uint64_t c = 0; c < m; ++c)
      EXPECT_EQ(pir.Retrieve(r, c), db[r * m + c])
          << "entry (" << r << "," << c << ")";
}

}  // namespace
}  // namespace primihub::pir::tiptoe

#else  // !PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoeLhePirTest, NeedsSeal) {
  GTEST_SKIP() << "build with --define=enable_tiptoe_real=1 + "
                  "--override_repository=underhood=<path> (needs SEAL toolchain)";
}

#endif
