/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * End-to-end YPIR (SimplePIR) retrieval at the paper's poly_len=2048 preset
 * (ParamsForExpansion) -- the scale-up from the v1 poly_len=8 binary-gadget
 * test. Validates the real-preset noise budget decodes exactly. Tagged
 * "manual" (heavy: 2048-pt NTT, a 2048x2048 db, recursive packing over 11
 * levels) so it is run explicitly, not in the default suite.
 */
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_chacha.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_client.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_discrete_gaussian.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_modulus_switch.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_packing.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_params.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_poly.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_regev.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_server.h"
#include "src/primihub/kernel/pir/operator/ypir/ypir_spiral_client.h"

namespace primihub::pir::ypir {
namespace {

const std::vector<std::uint64_t> kModuli = {268369921ULL, 249561089ULL};

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

// nu_1=0, nu_2=0, p=256, q2_bits=28, t_exp_left=2: the real poly_len=2048
// preset. db_rows = 2^(0+11) = 2048; db_cols = instances(1)*2048 = 2048.
TEST(YpirE2ELargeTest, SimplePirRetrievesRowPolyLen2048) {
  const Params p = ParamsForExpansion(/*nu_1=*/0, /*nu_2=*/0, /*p=*/256,
                                      /*q2_bits=*/28, /*t_exp_left=*/2, kModuli);
  ASSERT_EQ(p.poly_len, 2048u);
  NttContext ctx(p);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t db_cols = p.instances * p.poly_len;
  const std::size_t num_rlwe_outputs = db_cols / p.poly_len;

  std::vector<std::uint8_t> db(db_rows * db_cols);
  for (std::size_t r = 0; r < db_rows; ++r)
    for (std::size_t c = 0; c < db_cols; ++c)
      db[r * db_cols + c] =
          static_cast<std::uint8_t>((r * 31u + c * 17u + 3u) % 256u);
  const YServer<std::uint8_t> srv(p, db, /*is_simplepir=*/true,
                                  /*inp_transposed=*/false, /*pad_rows=*/false);

  ChaChaRng key = ChaChaRng::FromSeed(Seed(31));
  const Client client(ctx, /*hamming=*/256, key);
  ChaChaRng lwe_ent = ChaChaRng::FromSeed(Seed(32));
  const YClient yc(ctx, client, lwe_ent);

  const DiscreteGaussian dg = DiscreteGaussian::Init(p.noise_width);
  ChaChaRng ep = ChaChaRng::FromSeed(Seed(33));
  ChaChaRng ep_pub = ChaChaRng::FromSeed(Seed(34));
  const std::vector<PolyMatrixNTT> pack_pub_params = RawGenerateExpansionParams(
      ctx, dg, client.SkReg(), p.poly_len_log2, p.t_exp_left, ep, ep_pub);
  const YConstants y_constants = GenerateYConstants(ctx);

  const std::vector<std::uint64_t> hint_0 =
      srv.AnswerHintRing(ctx, kSeed0, db_cols);
  std::vector<std::uint64_t> combined = hint_0;
  combined.resize(hint_0.size() + db_cols, 0);
  const std::vector<std::vector<PolyMatrixNTT>> prepacked_lwe =
      PrepPackManyLwes(ctx, combined, num_rlwe_outputs);

  for (std::size_t target_row : {std::size_t{0}, std::size_t{1000},
                                 std::size_t{2047}}) {
    ChaChaRng noise = ChaChaRng::FromSeed(Seed(40));
    const std::vector<std::uint64_t> packed_query =
        yc.GenerateQueryPacked(kSeed0, p.db_dim_1, target_row, noise);
    const std::vector<std::uint64_t> intermediate =
        srv.AnswerQuery(packed_query);
    const std::vector<PolyMatrixNTT> packed =
        PackManyLwes(ctx, prepacked_lwe, intermediate, num_rlwe_outputs,
                     pack_pub_params, y_constants);

    std::vector<std::uint64_t> recovered;
    recovered.reserve(db_cols);
    for (const PolyMatrixNTT& ct : packed) {
      const PolyMatrixRaw dec = ctx.FromNtt(client.DecryptMatrixReg(ct));
      for (std::size_t z = 0; z < p.poly_len; ++z)
        recovered.push_back(Rescale(dec.data[z], p.modulus, p.pt_modulus));
    }

    const std::vector<std::uint8_t> expected_row = srv.GetRow(target_row);
    ASSERT_EQ(recovered.size(), expected_row.size());
    std::size_t mismatches = 0;
    for (std::size_t c = 0; c < db_cols; ++c)
      if (recovered[c] != static_cast<std::uint64_t>(expected_row[c]))
        ++mismatches;
    EXPECT_EQ(mismatches, 0u) << "target_row=" << target_row << " had "
                              << mismatches << " / " << db_cols << " mismatches";
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
