/*
 * Copyright (c) 2026 by PrimiHub
 * Licensed under the Apache License, Version 2.0
 *
 * End-to-end YPIR (SimplePIR) retrieval test (task 7.3 chunk 14). Wires the
 * full ported pipeline in-process:
 *   offline: AnswerHintRing(SEED_0) -> [hint_0, 0] -> PrepPackManyLwes
 *   online:  client GenerateQueryPacked(SEED_0) -> server AnswerQuery (AVX512
 *            kernel) -> PackManyLwes(prepacked, intermediate, pack_pub_params,
 *            y_constants) (recursive packing, replacing the precompute path)
 *   decode:  Client::DecryptMatrixReg + Rescale of each packed RLWE ct
 * and asserts the recovered row equals YServer::GetRow(target_row). Mod-switch
 * (switch/recover) is skipped -- it is only a wire-compression step and the
 * answer decodes from its raw form. P8b's binary expansion gadget keeps the
 * key-switch noise tiny so the small-parameter retrieval is exact.
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

constexpr std::uint64_t kQ0 = 268369921ull, kQ1 = 249561089ull;

// Binary expansion gadget (t_exp_left=60 -> bits_per=1): tiny key-switch noise.
Params P8b(std::size_t db_dim_1, std::size_t db_dim_2, std::size_t instances) {
  return Params::Init(8, {kQ0, kQ1}, 6.4, 1, 256, 28, 4, 60, 2, 3, true,
                      db_dim_1, db_dim_2, instances, 0, 0);
}

std::array<std::uint8_t, 32> Seed(std::uint8_t b) {
  std::array<std::uint8_t, 32> s{};
  for (auto& x : s) x = b;
  return s;
}

TEST(YpirE2ETest, SimplePirRetrievesRow) {
  // db_rows = 2^(1+3)=16; db_cols = instances*poly_len = 8; num_rlwe_outputs=1.
  const Params p = P8b(/*db_dim_1=*/1, /*db_dim_2=*/1, /*instances=*/1);
  NttContext ctx(p);
  const std::size_t db_rows = static_cast<std::size_t>(1)
                              << (p.db_dim_1 + p.poly_len_log2);
  const std::size_t db_cols = p.instances * p.poly_len;
  const std::size_t num_rlwe_outputs = db_cols / p.poly_len;

  // Plant a known db (values < pt_modulus); row-major input -> col-major store.
  std::vector<std::uint16_t> db(db_rows * db_cols);
  for (std::size_t r = 0; r < db_rows; ++r)
    for (std::size_t c = 0; c < db_cols; ++c)
      db[r * db_cols + c] =
          static_cast<std::uint16_t>((r * 11u + c * 7u + 3u) % 256u);
  const YServer<std::uint16_t> srv(p, db, /*is_simplepir=*/true,
                                   /*inp_transposed=*/false, /*pad_rows=*/false);

  // --- client setup: secret + packing expansion params ---
  ChaChaRng key = ChaChaRng::FromSeed(Seed(31));
  const Client client(ctx, /*hamming=*/2, key);
  ChaChaRng lwe_ent = ChaChaRng::FromSeed(Seed(32));
  const YClient yc(ctx, client, lwe_ent);

  const DiscreteGaussian dg = DiscreteGaussian::Init(p.noise_width);
  ChaChaRng ep = ChaChaRng::FromSeed(Seed(33));
  ChaChaRng ep_pub = ChaChaRng::FromSeed(Seed(34));
  const std::vector<PolyMatrixNTT> pack_pub_params = RawGenerateExpansionParams(
      ctx, dg, client.SkReg(), p.poly_len_log2, p.t_exp_left, ep, ep_pub);
  const YConstants y_constants = GenerateYConstants(ctx);

  // --- offline: hint -> prepacked ---
  const std::vector<std::uint64_t> hint_0 =
      srv.AnswerHintRing(ctx, kSeed0, db_cols);
  ASSERT_EQ(hint_0.size(), p.poly_len * db_cols);
  std::vector<std::uint64_t> combined = hint_0;
  combined.resize(hint_0.size() + db_cols, 0);  // append the zero b-row
  const std::vector<std::vector<PolyMatrixNTT>> prepacked_lwe =
      PrepPackManyLwes(ctx, combined, num_rlwe_outputs);

  // The offline hint/prepacked are target-independent; retrieve several rows.
  for (std::size_t target_row : {std::size_t{0}, std::size_t{3}, std::size_t{5},
                                 std::size_t{11}, std::size_t{15}}) {
    // --- online: query -> answer -> pack ---
    ChaChaRng noise =
        ChaChaRng::FromSeed(Seed(static_cast<std::uint8_t>(40 + target_row)));
    const std::vector<std::uint64_t> packed_query =
        yc.GenerateQueryPacked(kSeed0, p.db_dim_1, target_row, noise);
    ASSERT_EQ(packed_query.size(), db_rows);
    const std::vector<std::uint64_t> intermediate =
        srv.AnswerQuery(packed_query);
    ASSERT_EQ(intermediate.size(), db_cols);
    const std::vector<PolyMatrixNTT> packed =
        PackManyLwes(ctx, prepacked_lwe, intermediate, num_rlwe_outputs,
                     pack_pub_params, y_constants);
    ASSERT_EQ(packed.size(), num_rlwe_outputs);

    // --- decode: DecryptMatrixReg + rescale ---
    std::vector<std::uint64_t> recovered;
    recovered.reserve(db_cols);
    for (const PolyMatrixNTT& ct : packed) {
      const PolyMatrixRaw dec = ctx.FromNtt(client.DecryptMatrixReg(ct));
      for (std::size_t z = 0; z < p.poly_len; ++z)
        recovered.push_back(Rescale(dec.data[z], p.modulus, p.pt_modulus));
    }

    const std::vector<std::uint16_t> expected_row = srv.GetRow(target_row);
    ASSERT_EQ(recovered.size(), expected_row.size());
    for (std::size_t c = 0; c < db_cols; ++c)
      EXPECT_EQ(recovered[c], static_cast<std::uint64_t>(expected_row[c]))
          << "target_row=" << target_row << " col=" << c;
  }
}

}  // namespace
}  // namespace primihub::pir::ypir
