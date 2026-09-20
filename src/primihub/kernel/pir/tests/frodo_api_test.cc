/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * frodo_api_test — chunk 5 verification. Covers the four api.rs
 * types individually + a full PIR end-to-end roundtrip that seals
 * the algorithmic loop: 4-element DB, client queries index k,
 * recovers original bytes byte-for-byte.
 */
#include "src/primihub/kernel/pir/operator/frodo_pir/frodo_api.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "base64.h"  // NOLINT

namespace primihub::pir::frodo {
namespace {

SeedBytes IotaSeed(std::uint8_t start) {
  SeedBytes s;
  for (std::size_t i = 0; i < 32; ++i) {
    s[i] = static_cast<std::uint8_t>(start + i);
  }
  return s;
}

std::string B64(const std::vector<std::uint8_t>& bytes) {
  std::string s(bytes.begin(), bytes.end());
  return base64_encode(
      reinterpret_cast<const unsigned char*>(s.data()), s.size());
}

// Generates m random 1-byte DB elements seeded by a fixed mt19937
// for test reproducibility. Element i = byte (i * 17 + 13) mod 256
// — picked to avoid the all-same-byte degenerate case.
std::vector<std::string> MakeDeterministicDbElems(std::size_t m) {
  std::vector<std::string> elems;
  elems.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    const std::uint8_t b =
        static_cast<std::uint8_t>((i * 17u + 13u) & 0xFFu);
    elems.push_back(B64({b}));
  }
  return elems;
}

// ---- Query + Response trivial wrappers -------------------------

TEST(FrodoQueryTest, AsSlice_RoundtripsCtorArg) {
  const std::vector<std::uint32_t> v = {1u, 2u, 3u};
  const Query q(v);
  EXPECT_EQ(q.AsSlice(), v);
}

TEST(FrodoResponseTest, AsSlice_RoundtripsCtorArg) {
  const std::vector<std::uint32_t> v = {100u, 200u};
  const Response r(v);
  EXPECT_EQ(r.AsSlice(), v);
}

// ---- QueryParams ----------------------------------------------

TEST(FrodoQueryParamsTest, New_SizesMatchBaseParams) {
  // Build a tiny shard and check QueryParams.lhs.size() = m,
  // QueryParams.rhs.size() = db.matrix_width_self().
  const std::size_t m = 4;
  const std::size_t dim = 6;
  const std::size_t elem_size = 8;
  const std::size_t plaintext_bits = 5;
  const auto elems = MakeDeterministicDbElems(m);
  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, dim, m, elem_size,
                                              plaintext_bits, IotaSeed(0),
                                              &shard, &err),
            retcode::SUCCESS) << err;
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);
  QueryParams qp;
  ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(qp.GetLhs().size(), m);
  EXPECT_EQ(qp.GetRhs().size(),
            Database::GetMatrixWidth(elem_size, plaintext_bits));
  EXPECT_EQ(qp.GetElemSize(), elem_size);
  EXPECT_EQ(qp.GetPlaintextBits(), plaintext_bits);
  EXPECT_FALSE(qp.IsUsed());
}

TEST(FrodoQueryParamsTest, GenerateQuery_FlipsUsedFlag) {
  const std::size_t m = 4;
  const std::size_t dim = 6;
  const auto elems = MakeDeterministicDbElems(m);
  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, dim, m, 8, 5,
                                              IotaSeed(1), &shard, &err),
            retcode::SUCCESS) << err;
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);
  QueryParams qp;
  ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS);
  Query q;
  ASSERT_EQ(qp.GenerateQuery(0, &q, &err), retcode::SUCCESS) << err;
  EXPECT_TRUE(qp.IsUsed());
  EXPECT_EQ(q.AsSlice().size(), m);
}

TEST(FrodoQueryParamsTest, GenerateQuery_ReuseRejected) {
  const auto elems = MakeDeterministicDbElems(4);
  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, 6, 4, 8, 5,
                                              IotaSeed(2), &shard, &err),
            retcode::SUCCESS);
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);
  QueryParams qp;
  ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS);
  Query q1;
  ASSERT_EQ(qp.GenerateQuery(0, &q1, &err), retcode::SUCCESS);
  // Second call must fail with the upstream-named diagnostic.
  Query q2;
  EXPECT_EQ(qp.GenerateQuery(0, &q2, &err), retcode::FAIL);
  EXPECT_NE(err.find("ErrorQueryParamsReused"), std::string::npos) << err;
}

TEST(FrodoQueryParamsTest, GenerateQuery_RowIndexOutOfRange) {
  const auto elems = MakeDeterministicDbElems(4);
  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, 6, 4, 8, 5,
                                              IotaSeed(3), &shard, &err),
            retcode::SUCCESS);
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);
  QueryParams qp;
  ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS);
  Query q;
  EXPECT_EQ(qp.GenerateQuery(99, &q, &err), retcode::FAIL);
  EXPECT_NE(err.find("out of range"), std::string::npos) << err;
}

// ---- Shard -----------------------------------------------------

TEST(FrodoShardTest, FromBase64StringsWithSeed_ReproducibleSetup) {
  const auto elems = MakeDeterministicDbElems(4);
  const auto seed = IotaSeed(7);
  Shard a, b;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, 6, 4, 8, 5,
                                              seed, &a, &err),
            retcode::SUCCESS) << err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, 6, 4, 8, 5,
                                              seed, &b, &err),
            retcode::SUCCESS) << err;
  EXPECT_EQ(a.GetBaseParams().GetPublicSeed(),
            b.GetBaseParams().GetPublicSeed());
  EXPECT_EQ(a.GetBaseParams().RhsFlat(),
            b.GetBaseParams().RhsFlat());
}

TEST(FrodoShardTest, Respond_LengthMatchesDbWidth) {
  const auto elems = MakeDeterministicDbElems(4);
  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, 6, 4, 8, 5,
                                              IotaSeed(8), &shard, &err),
            retcode::SUCCESS);
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);
  QueryParams qp;
  ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS);
  Query q;
  ASSERT_EQ(qp.GenerateQuery(2, &q, &err), retcode::SUCCESS);
  Response resp;
  ASSERT_EQ(shard.Respond(q, &resp, &err), retcode::SUCCESS) << err;
  EXPECT_EQ(resp.AsSlice().size(),
            shard.GetDb().GetMatrixWidthSelf());
}

// ---- THE END-TO-END ROUNDTRIP ---------------------------------

// The flagship test: do a real FrodoPIR retrieval. 4-element DB,
// query each index, verify the recovered bytes match the original.
// If this passes, the entire LWE+PIR pipeline composes correctly.
TEST(FrodoApiE2ETest, RoundtripAllIndices_4Element) {
  const std::size_t m = 4;
  const std::size_t dim = 16;
  const std::size_t elem_size = 8;     // 1 byte
  const std::size_t plaintext_bits = 5;  // matches chunk 3c clean params

  // Build DB with hand-picked distinct bytes so each retrieval
  // is unambiguously "the right one".
  const std::vector<std::uint8_t> orig_bytes = {0x12u, 0x34u, 0x56u, 0x78u};
  std::vector<std::string> elems;
  for (auto b : orig_bytes) elems.push_back(B64({b}));

  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, dim, m, elem_size,
                                              plaintext_bits, IotaSeed(0x42),
                                              &shard, &err),
            retcode::SUCCESS) << err;
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);

  for (std::size_t k = 0; k < m; ++k) {
    // Fresh QueryParams per retrieval — single-use enforcement.
    QueryParams qp;
    ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS) << err;
    Query q;
    ASSERT_EQ(qp.GenerateQuery(k, &q, &err), retcode::SUCCESS) << err;

    Response resp;
    ASSERT_EQ(shard.Respond(q, &resp, &err), retcode::SUCCESS) << err;

    const auto got_bytes = resp.ParseOutputAsBytes(qp);
    ASSERT_EQ(got_bytes.size(), 1u) << "elem " << k;
    EXPECT_EQ(got_bytes[0], orig_bytes[k])
        << "PIR retrieved wrong byte at index " << k
        << ": got " << static_cast<int>(got_bytes[0])
        << ", expected " << static_cast<int>(orig_bytes[k]);

    // Also verify base64 path matches.
    const auto got_b64 = resp.ParseOutputAsBase64(qp);
    EXPECT_EQ(got_b64, elems[k])
        << "PIR base64 retrieval at index " << k
        << ": got '" << got_b64 << "', expected '" << elems[k] << "'";
  }
}

// Same end-to-end, larger DB, randomized contents — exercises
// distinct fixture from the hand-picked one.
TEST(FrodoApiE2ETest, RoundtripRandomElements_16) {
  const std::size_t m = 16;
  const std::size_t dim = 32;
  const std::size_t elem_size = 8;
  const std::size_t plaintext_bits = 5;
  const auto elems = MakeDeterministicDbElems(m);

  Shard shard;
  std::string err;
  ASSERT_EQ(Shard::FromBase64StringsWithSeed(elems, dim, m, elem_size,
                                              plaintext_bits, IotaSeed(99),
                                              &shard, &err),
            retcode::SUCCESS) << err;
  const auto& bp = shard.GetBaseParams();
  const auto cp = CommonParams::FromBaseParams(bp);

  // Test indices 0, 5, 15 (start / middle / end).
  for (std::size_t k : {std::size_t{0}, std::size_t{5}, std::size_t{15}}) {
    QueryParams qp;
    ASSERT_EQ(QueryParams::New(cp, bp, &qp, &err), retcode::SUCCESS);
    Query q;
    ASSERT_EQ(qp.GenerateQuery(k, &q, &err), retcode::SUCCESS);
    Response resp;
    ASSERT_EQ(shard.Respond(q, &resp, &err), retcode::SUCCESS);
    const auto got_b64 = resp.ParseOutputAsBase64(qp);
    EXPECT_EQ(got_b64, elems[k]) << "index " << k;
  }
}

}  // namespace
}  // namespace primihub::pir::frodo
