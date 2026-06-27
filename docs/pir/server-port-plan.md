# YPIR server.rs (chunk 10) — C++ port plan

Upstream `ypir/src/server.rs` is 1,246 LOC. Unlike the lower chunks,
**server.rs sits near the TOP of the YPIR dependency graph** — the
14-chunk plan (ypir-port-plan.md) listed it before scheme.rs/client.rs,
but in reality server.rs *calls into* scheme, client, the large packing
algorithms, and the AVX512 kernel. So it cannot be ported wholesale yet;
it is decomposed into sub-chunks gated on their real dependencies.

## Dependency reality (what server.rs actually needs)

| server.rs symbol | comes from | C++ status |
|---|---|---|
| `generate_y_constants`, `split_alloc`, `db_rows_padded`, `db_cols` | self (leaf) | **DONE (10a)** |
| `write_bits`/`read_bits` | bits.rs | ported (ypir_bits) |
| `to_ntt`/`multiply`/`add_into`/`add_into_no_reduce` | spiral poly | ported (ypir_poly / ypir_poly_ops) |
| `transpose_generic` | transpose.rs | ported (ypir_transpose) |
| `matmul_vec_packed` | matmul.rs | ported (ypir_matmul, vendored) |
| `Convolution`, `negacyclic_perm_u32`, `naive_multiply_matrices` | convolution.rs | ported (ypir_convolution[_ntt]) |
| `barrett_coeff_u64`, `log2`, `rescale` | arith / modulus_switch | ported (ypir_arith / ypir_modulus_switch) |
| `SEED_0/1`, `get_seed`, `STATIC_SEED_2`, `YPIRParams`, `get_q_prime_*` | **scheme.rs (chunk 11)** | NOT ported |
| `Client`, `YClient`, `generate_query_impl`, `generate_matrix_ring` | **client.rs (chunk 12)** | partial (lwe_client/regev exist; YClient query gen does not) |
| `prep_pack_many_lwes`, `pack_many_lwes`, `precompute_pack` | **packing.rs (chunk 9 tail)** | NOT ported (only fast-ops + condense exist) |
| `fast_batched_dot_product_avx512` | **kernel.rs (chunk 5)** | NOT ported; needs AVX512 host (.50 is Broadwell) |
| `AlignedMemory64` | spiral_rs | use std::vector<uint64_t> (8-byte aligned enough) or a thin aligned alloc |

## Sub-chunks

- **10a — leaf functions (DONE).** `GenerateYConstants` / `SplitAlloc` /
  `DbRowsPadded` / `DbCols` in `ypir_server.{h,cc}` + `ypir_server_test`
  (4 cases, manual/HEXL). Oracles: monomial round-trip via FromNtt;
  hand-computed bitstream re-chunk; dimension arithmetic. PASS.
- **10b — YServer<T> struct + new() + db accessors.** DB layout into an
  aligned buffer (row/col transposed indexing), smaller_params
  derivation (DoublePIR round), `db()/db_u16()/db_u32()/get_elem/get_row`.
  Template over T ∈ {u8,u16,u32} (ToU64). Oracle: round-trip a small DB
  through new()+get_elem with both transposed/non-transposed layouts and
  both is_simplepir modes. No crypto deps — portable now.
- **10c — multiply_with_db_ring + generate_hint_0_ring.** Ring (RLWE)
  matmul over the DB using ported to_ntt/multiply/add_into + Convolution.
  Oracle: compare against naive_multiply_matrices on a small DB.
  Depends only on ported primitives — portable now (after 10b).
  generate_hint_0 (non-ring) uses naive_multiply_matrices directly.
- **10d — generate_pseudorandom_query + answer_hint_ring.** BLOCKED on
  chunk 12 YClient::generate_query_impl + scheme SEED_* / get_seed.
- **10e — answer_query / multiply_batched_with_db_packed / online
  first pass.** BLOCKED on chunk 5 AVX512 kernel (or a scalar fallback
  port; upstream has a `__m512i = u64` non-avx512 module that yields a
  scalar reference path we can mirror, like packing's FastMultiplyNoReduce).
- **10f — perform_offline_precomputation[_simplepir].** BLOCKED on chunk
  9 tail (prep_pack_many_lwes / precompute_pack) + 10b-e.
- **10g — perform_online_computation[_simplepir]<K>.** BLOCKED on chunk
  9 tail (pack_many_lwes) + 10e-f. This is the full Answer pipeline.

## Notes
- All HEXL-touching tests are tagged `manual`.
- Build/test MUST pass `--config=linux_x86_64` (sets -std=c++17; without
  it std::make_unique in ypir_poly.cc fails to compile at the default
  c++11). Fetches go through mihomo 7890 (see .bazelrc repo_env).
- The honest critical path to a working YPIR operator is: chunk 11
  (scheme) + finish chunk 12 (YClient) + chunk 9 tail (pack_many_lwes) +
  a scalar fast_batched_dot_product → then 10d/10e/10f/10g wire up.
