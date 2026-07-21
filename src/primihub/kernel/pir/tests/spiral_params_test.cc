/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * EstimateParams unit tests. Cover the canonical-record case (matches the
 * upstream "wiki" parameter table entry within the published band), the
 * minimum / cap rejection paths, and the consistency invariant
 * total_n == 2^(nu_1 + nu_2).
 */
#include <gtest/gtest.h>

#include <string>

#include "src/primihub/kernel/pir/operator/spiral_pir/params.h"

namespace primihub::pir::spiral {

TEST(SpiralParamsTest, OneMillionRecordsLandsInWikiBand) {
  // Upstream's canonical "wiki" config is (nu_1=9, nu_2=11) at 2^20 = ~1M
  // records. Our balanced split gives (10, 10); both produce total_n=2^20
  // and the same 14 KB query-size class.
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(1'000'000, 256, &p, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(p.nu_1 + p.nu_2, 20u);
  EXPECT_EQ(p.total_n, 1ULL << 20);
}

TEST(SpiralParamsTest, SingleRecordClampsToMinNu) {
  // 1 record needs only 1 bit of addressing, but kMinNu=4 means both nu_1
  // and nu_2 floor at 4; total_n = 2^8 = 256 (wasteful but legal).
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(1, 8, &p, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(p.nu_1, kMinNu);
  EXPECT_EQ(p.nu_2, kMinNu);
  EXPECT_EQ(p.total_n, 1ULL << (kMinNu * 2));
}

TEST(SpiralParamsTest, MediumDbBalancedSplit) {
  // 65536 records = 2^16 → balanced split (8, 8); 1024-byte records OK.
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(65'536, 1024, &p, &err), retcode::SUCCESS) << err;
  EXPECT_GE(p.nu_1 + p.nu_2, 16u);  // covers 2^16
  EXPECT_EQ(p.total_n, 1ULL << (p.nu_1 + p.nu_2));
  // Selection rule biases toward nu_1 (= ceil(total_bits/2)).
  EXPECT_GE(p.nu_1, p.nu_2);
}

TEST(SpiralParamsTest, ZeroRecordsRejected) {
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(0, 256, &p, &err), retcode::FAIL);
  EXPECT_NE(err.find("num_records"), std::string::npos)
      << "error message should name the offending input; got: " << err;
}

TEST(SpiralParamsTest, OverCapRejectedWithGuidance) {
  // 2M records — past kMaxRecords; error string must point at the v1 cap
  // and mention the SpiralStream follow-up so log readers know the fix.
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(2'000'000, 256, &p, &err), retcode::FAIL);
  EXPECT_NE(err.find("1M"), std::string::npos) << err;
  EXPECT_NE(err.find("SpiralStream"), std::string::npos) << err;
}

TEST(SpiralParamsTest, OversizedRecordRejected) {
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(1024, kMaxRecordBytes + 1, &p, &err), retcode::FAIL);
  EXPECT_NE(err.find("2048"), std::string::npos) << err;
}

TEST(SpiralParamsTest, ZeroRecordBytesRejected) {
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(100, 0, &p, &err), retcode::FAIL);
  EXPECT_NE(err.find("record_size_bytes"), std::string::npos) << err;
}

TEST(SpiralParamsTest, NullOutParamsRejected) {
  SpiralParams p;
  std::string err;
  EXPECT_EQ(EstimateParams(100, 100, nullptr, &err), retcode::FAIL);
  EXPECT_EQ(EstimateParams(100, 100, &p, nullptr), retcode::FAIL);
}

TEST(SpiralParamsTest, ParamsBoundsRespected) {
  // Sweep the full legal record-count range; every selection must respect
  // [kMinNu, kMaxNu1] x [kMinNu, kMaxNu2] and total_n must dominate the input.
  const uint64_t samples[] = {1ULL, 100ULL, 1000ULL, 10000ULL, 100000ULL, 500000ULL, kMaxRecords};
  for (uint64_t n : samples) {
                     
    SpiralParams p;
    std::string err;
    ASSERT_EQ(EstimateParams(n, 256, &p, &err), retcode::SUCCESS) << n;
    EXPECT_GE(p.nu_1, kMinNu) << n;
    EXPECT_LE(p.nu_1, kMaxNu1) << n;
    EXPECT_GE(p.nu_2, kMinNu) << n;
    EXPECT_LE(p.nu_2, kMaxNu2) << n;
    EXPECT_GE(p.total_n, n) << n;
    EXPECT_EQ(p.total_n, 1ULL << (p.nu_1 + p.nu_2)) << n;
  }
}

}  // namespace primihub::pir::spiral
