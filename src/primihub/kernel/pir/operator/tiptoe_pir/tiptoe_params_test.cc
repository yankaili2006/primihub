/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * Test for the rlwe Context RAII wrapper (tiptoe chunk 1.1c). Real mode needs
 * SEAL (GFW-blocked on .50), so this is manual + gated on
 * --define=enable_tiptoe_real=1; without it compiles to a single skipped test.
 * The same checks were validated standalone against SEAL 4.1 when this chunk
 * landed (N=2048 P=65537 LogQ=38).
 */
#include "gtest/gtest.h"

#ifdef PIR_TIPTOE_RLWE_VENDORED

#include <utility>

#include "src/primihub/kernel/pir/operator/tiptoe_pir/tiptoe_params.h"

namespace primihub::pir::tiptoe {
namespace {

TEST(TiptoeParamsTest, ContextParams) {
  Params p;
  EXPECT_EQ(p.N(), 2048u);
  EXPECT_EQ(p.P(), 65537u);
  EXPECT_EQ(p.LogQ(), 38u);
  EXPECT_NE(p.ctx(), nullptr);
}

TEST(TiptoeParamsTest, MoveTransfersOwnership) {
  Params p;
  Params q = std::move(p);  // no double-free at scope end
  EXPECT_EQ(q.N(), 2048u);
  EXPECT_EQ(q.P(), 65537u);
}

}  // namespace
}  // namespace primihub::pir::tiptoe

#else  // !PIR_TIPTOE_RLWE_VENDORED

TEST(TiptoeParamsTest, NeedsSeal) {
  GTEST_SKIP() << "build with --define=enable_tiptoe_real=1 + "
                  "--override_repository=underhood=<path> (needs SEAL toolchain)";
}

#endif
