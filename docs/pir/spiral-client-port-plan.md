# Spiral RLWE Client Port — Plan (YPIR task 7.3, "chunk 12 pt2")

## Scope reality check

The "spiral RLWE Client" blocker is **not** the full spiral Phase-3 runtime port
(the `Params`/`PolyMatrixRaw`/`PolyMatrixNTT` runtime + GSW + packing machinery
that blocks the *other* upper YPIR layers per design.md D10). YPIR query
generation needs only the **Regev (RLWE) encryption surface** of
`spiral_rs::client::Client`, which is ~120 LOC and maps almost 1:1 onto
primitives already living in `ypir_regev`.

Confirmed by reading spiral-rs@6929441 `src/client.rs`:
`encrypt_matrix_reg` / `encrypt_matrix_scaled_reg` are thin compositions over
`get_fresh_reg_public_key` (= ported `GetFreshRegPublicKey`) + `PolyMatrix.pad_top(1)`
(= ported `PadTopNtt`) + `+` (= ported `AddNtt`). The only genuinely new pieces
are the **scaled** sample variant, the **ternary secret**, and a small `Client`
holder.

## What `YClient::generate_query_impl` + `decode_response` need from `Client`

- `Client` holds `sk_reg` (ternary, Hamming weight 256) + `dg` (DiscreteGaussian).
- `get_sk_reg()`
- `encrypt_matrix_reg(a_ntt, rng, rng_pub)`
- `encrypt_matrix_scaled_reg(a_ntt, rng, rng_pub, scale)`
- decode uses `sk_reg` directly in `YClient::decode_response`'s phase sum
  (`decrypt_matrix_reg` optional, only for unit round-trip tests).

GSW (`sk_gsw`, `encrypt_matrix_gsw`) is NOT needed for query generation — skip.

## Have vs. Need

**Already ported** (`ypir_regev` / `ypir_poly_ops` / `ypir_poly`):
`GetRegSample`, `GetFreshRegPublicKey`, `RawGenerateExpansionParams`
(= `generate_expansion_params`), `RegevDecrypt`, `PadTopNtt`, `AddNtt`,
`NoiseRaw`, `RandomRngRaw`, `ToNtt`/`FromNtt`, `ScalarMultiplyNtt`.

**Need to port** (small):
1. `gen_ternary_mat` + `HAMMING_WEIGHT=256` (ternary secret) — or reuse a
   `NoiseRaw` secret (see decision).
2. `matrix_with_identity` (`sk_reg_full = [sk_reg; I]`) — only for
   `decrypt_matrix_reg` round-trip tests.
3. `GetScaledRegevSample` (= `GetRegSample` but each noise coeff
   `e[i] = multiply_uint_mod(e[i], scale, modulus)` before adding).
4. `GetFreshScaledRegPublicKey` (m columns of the scaled sample).
5. `EncryptMatrixReg(a)` = `GetFreshRegPublicKey(m) + PadTopNtt(a, 1)`.
6. `EncryptMatrixScaledReg(a, scale)` = `GetFreshScaledRegPublicKey(m, scale) + PadTopNtt(a, 1)`.
7. Verify/port arith helpers: `multiply_uint_mod`, `invert_uint_mod`,
   `single_value`/`single_poly` (several already exist — confirm in 12b-1).

## Key decision: secret key — ternary (faithful) vs. Gaussian (pragmatic)

YPIR is single-impl (no cross-impl wire), so the secret only needs internal
consistency + security.

- **(A) Faithful ternary** (`gen_ternary_mat`, HW=256): matches spiral's noise
  budget exactly; keeps `noise_analysis` valid at scale; ~15 LOC.
- **(B) Gaussian secret** (`NoiseRaw`): already proven in `ypir_regev_test`;
  lower risk now, but noise budget diverges from spiral's analysis (may bite at
  large poly_len/db).

**Recommendation: (A) ternary.** It's cheap and keeps the documented noise
analysis honest. Keep the `Client` API so a test can pin encrypt/decrypt
round-trip either way.

## Chunk breakdown

- **12b-1 — arith/helper gaps.** Verify/port `multiply_uint_mod`,
  `invert_uint_mod`, `single_value`, `matrix_with_identity`, `gen_ternary_mat`
  (+`HAMMING_WEIGHT`). Per-helper unit tests (ternary Hamming weight exact;
  `matrix_with_identity` shape; `multiply_uint_mod`/`invert_uint_mod` vs
  hand-computed). ~1 session.
- **12b-2 — `Client` + Reg encryption.** `ypir_spiral_client.{h,cc}`: `Client`
  (ternary `sk_reg`, `dg`, `GetSkReg`) + `GetScaledRegevSample` +
  `GetFreshScaledRegPublicKey` + `EncryptMatrixReg` + `EncryptMatrixScaledReg`
  + `DecryptMatrixReg`. Oracle: `EncryptMatrixReg(a)` → `DecryptMatrixReg`
  round-trips a known NTT matrix (mirrors `ypir_regev_test`); scaled variant
  decrypts to the scale-adjusted plaintext. ~1 session.
- **12b-3 — `YClient::generate_query_impl` + `decode_response`.** Build on
  `EncryptMatrixScaledReg` + `invert_uint_mod(poly_len)` + `single_value` +
  `ScalarMultiplyNtt`; `rlwes_to_lwes` (`rlwe_to_lwe_last_row` + `concat_horizontal`,
  mostly ported). **Oracle = first true E2E**: tiny `YServer`, `generate_query`
  (packing) → `MultiplyWithDbRing`/answer → `decode_response` recovers the
  planted DB element exactly. ~1–2 sessions.
- **12b-4 — `generate_query` SEED_0 LWE branch.** The SimplePIR first-dimension
  query via `LweEncryptMany`. Oracle: decode via the LWE secret. ~1 session.

## Test strategy

Per-primitive unit tests (12b-1) → Client encrypt/decrypt round-trip (12b-2,
`ypir_regev_test` style) → **E2E retrieval (12b-3)**: plant a value in a small
DB, run query → answer → decode, assert exact recovery. 12b-3 is the first
genuine end-to-end YPIR correctness signal and de-risks the whole online path.

## After this lands

Unblocks server **10d** (`generate_pseudorandom_query` + `answer_hint_ring`),
then **10e** (`answer_query` needs the AVX512 `fast_batched_dot_product` — do on
the local AVX512 host), **10f/10g**, and finally `YpirOperator::OnExecute`
(`kIsSkeleton=false`, `caps.is_real=true`) + the 64-query E2E test (chunk 14) —
making YPIR the 5th real PIR algorithm.

The **Convolution-params / `barrett_coeff_u64` / `log2`** gap (for
`generate_hint_0_ring` / `answer_hint_ring`) is independent and can proceed in
parallel as its own track.

## Risks

- **Noise budget** if Gaussian secret chosen → mitigated by ternary (option A).
- **RNG consumption order**: client encrypt draws `a` from `rng_pub`; the
  server reconstructs the same public randomness. Mirror `GetRegSample`'s order
  exactly (already established + tested in `ypir_regev_test`).
- **`scale` semantics**: `multiply_uint_mod` on each noise coeff — pin with a
  dedicated test in 12b-2.
- **Ternary RNG draw order** in `gen_ternary_mat` must match upstream so the
  secret distribution/budget holds — unit-test the Hamming weight + determinism.
