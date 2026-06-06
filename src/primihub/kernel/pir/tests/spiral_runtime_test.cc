/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * SpiralRuntime end-to-end tests. Behaviour depends on
 * PIR_SPIRAL_RUNTIME_VENDORED:
 *   * unset: every facade method returns FAIL with "not vendored" — verify
 *     that contract so the stub-mode build is observable.
 *   * set (--define=enable_spiral_real=1): EnsureInitialized must run
 *     upstream setup_constants/generate_gadgets/build_table without
 *     crashing and lock the params; ClientEncode/ServerProcess/ClientDecode
 *     still return FAIL with "not yet implemented" until the followup commit.
 *
 * The split lets CI exercise both branches without forking the test files.
 */
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "src/primihub/kernel/pir/operator/spiral_pir/params.h"
#include "src/primihub/kernel/pir/operator/spiral_pir/spiral_runtime.h"

namespace primihub::pir::spiral {

TEST(SpiralRuntimeTest, VendoredFlagAgreesWithBuild) {
#ifdef PIR_SPIRAL_RUNTIME_VENDORED
  EXPECT_TRUE(kSpiralRuntimeVendored)
      << "PIR_SPIRAL_RUNTIME_VENDORED is set but kSpiralRuntimeVendored=false";
#else
  EXPECT_FALSE(kSpiralRuntimeVendored)
      << "PIR_SPIRAL_RUNTIME_VENDORED is unset but kSpiralRuntimeVendored=true";
#endif
}

TEST(SpiralRuntimeTest, EnsureInitializedBehavesPerMode) {
  // Note: SpiralRuntime is a process-wide singleton, so this test must be
  // the FIRST one to touch it (gtest runs in test-order; we picked a name
  // that sorts after the trivial Vendored check).
  SpiralParams p{};
  std::string err;
  ASSERT_EQ(EstimateParams(1024, 64, &p, &err), retcode::SUCCESS) << err;

  auto rc = SpiralRuntime::Instance().EnsureInitialized(p, &err);
  if (kSpiralRuntimeVendored) {
    EXPECT_EQ(rc, retcode::SUCCESS) << "real-mode EnsureInitialized failed: "
                                    << err;
    auto locked = SpiralRuntime::Instance().locked_params();
    EXPECT_EQ(locked.nu_1, p.nu_1);
    EXPECT_EQ(locked.nu_2, p.nu_2);
    EXPECT_EQ(locked.total_n, p.total_n);
  } else {
    EXPECT_EQ(rc, retcode::FAIL);
    EXPECT_NE(err.find("not vendored"), std::string::npos) << err;
  }
}

TEST(SpiralRuntimeTest, IdempotentOnSameParams) {
  SpiralParams p{};
  std::string err;
  ASSERT_EQ(EstimateParams(1024, 64, &p, &err), retcode::SUCCESS);

  auto& rt = SpiralRuntime::Instance();
  auto rc1 = rt.EnsureInitialized(p, &err);
  auto rc2 = rt.EnsureInitialized(p, &err);

  if (kSpiralRuntimeVendored) {
    EXPECT_EQ(rc1, retcode::SUCCESS);
    EXPECT_EQ(rc2, retcode::SUCCESS) << "second call on same params should "
                                       "be idempotent: " << err;
  } else {
    // stub mode: both fail with same message
    EXPECT_EQ(rc1, retcode::FAIL);
    EXPECT_EQ(rc2, retcode::FAIL);
  }
}

TEST(SpiralRuntimeTest, RejectsConflictingParamsInRealMode) {
  if (!kSpiralRuntimeVendored) GTEST_SKIP() << "stub mode";

  // First init from previous test already locked the singleton to
  // (1024, 64)'s params. Try a different params and confirm rejection.
  SpiralParams p_first{}, p_other{};
  std::string err;
  ASSERT_EQ(EstimateParams(1024, 64, &p_first, &err), retcode::SUCCESS);
  ASSERT_EQ(EstimateParams(500'000, 64, &p_other, &err), retcode::SUCCESS);
  ASSERT_NE(p_first.nu_1, p_other.nu_1)
      << "test setup invariant violated: chosen sizes must yield distinct "
         "(nu_1, nu_2) — adjust if EstimateParams selection changes";

  auto& rt = SpiralRuntime::Instance();
  ASSERT_EQ(rt.EnsureInitialized(p_first, &err), retcode::SUCCESS);
  auto rc = rt.EnsureInitialized(p_other, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  EXPECT_NE(err.find("single-params-per-process"), std::string::npos) << err;
}

TEST(SpiralRuntimeTest, ClientEncodeReturnsArchitecturalBlocker) {
  // v1 finding (verified empirically 2026-06-06): upstream Spiral cannot be
  // cleanly split into client-encode / server-process roles because
  // generate_setup at spiral.cpp L1546 leaves g_Ws_fft unwritten unless
  // direct_upload=true, and the wiki config we picked has direct_upload
  // off. ClientEncode therefore returns FAIL with a refactor-pointer
  // message in BOTH stub and real modes — the same outcome for different
  // reasons, but the user-facing contract is identical: don't depend on
  // ClientEncode in v1.
  SpiralParams p{};
  std::string err;
  ASSERT_EQ(EstimateParams(1024, 64, &p, &err), retcode::SUCCESS);
  auto& rt = SpiralRuntime::Instance();
  rt.EnsureInitialized(p, &err);  // success real / fail stub — both fine

  std::vector<uint8_t> blob;
  auto rc = rt.ClientEncode(0, &blob, &err);
  EXPECT_EQ(rc, retcode::FAIL);
  if (kSpiralRuntimeVendored) {
    EXPECT_NE(err.find("upstream refactor"), std::string::npos) << err;
    EXPECT_NE(err.find("single-process"), std::string::npos) << err;
  } else {
    EXPECT_NE(err.find("not vendored"), std::string::npos) << err;
  }
}

TEST(SpiralRuntimeTest, ServerAndDecodeStillTodo) {
  // ServerProcess + ClientDecode remain stubs pending the multi-day crypto
  // refactor (see commit message of e2248a12 / 21a73ad6).
  std::vector<uint8_t> in, out;
  std::string val, err;
  auto& rt = SpiralRuntime::Instance();
  EXPECT_EQ(rt.ServerProcess(in, std::vector<std::string>{"a"}, &out, &err),
            retcode::FAIL);
  EXPECT_EQ(rt.ClientDecode(out, &val, &err), retcode::FAIL);
}

}  // namespace primihub::pir::spiral

#ifdef PIR_SPIRAL_RUNTIME_VENDORED
// Custom main only in real mode: upstream Spiral allocates many globals via
// malloc/aligned_alloc with no matching frees (load_db/setup_constants both
// leak by design — the upstream binary's lifecycle ends at process exit so
// the OS reclaims). When primihub's gtest dtors run after our singleton
// holds those allocations, the interaction triggers double-free aborts at
// process teardown. Skip C++ static dtors via _exit(rc) so the test process
// reports correct exit status to bazel. In stub mode no upstream allocation
// runs and gtest_main's default works fine.
#include <cstdlib>
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  std::fflush(stdout);
  std::fflush(stderr);
  _exit(rc);  // intentional — see comment above
}
#endif
