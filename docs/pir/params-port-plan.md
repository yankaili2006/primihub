# YPIR spiral_rs Params / PolyMatrix Port — Plan (prerequisite for chunks 7+)

## Why

Everything in YPIR above the LWE layer (params.rs, packing.rs, server.rs,
scheme.rs, client.rs) builds on spiral_rs's **runtime** `Params` and
`PolyMatrixRaw`/`PolyMatrixNTT`. Convolution (chunk 6) slipped through
because it used HEXL's `NTT(n,q)` *directly*, bypassing `Params`. Nothing
above it can: `params.rs::ext_params_from_json` calls
`Params::init(poly_len, &moduli, …, n, p, q2_bits, t_conv, t_exp_*, t_gsw,
db_dim_1, db_dim_2, …)` to build a per-instance runtime config.

primihub's vendored C++ Spiral (`@spiral_pir`) has a **compile-time**
`Params` (fixed by `SPIRAL_DEFINES`, the "wiki" 1M config) and a global
`MatPoly`. It cannot represent YPIR's runtime params. So the prerequisite
is a **standalone C++ port of spiral_rs's runtime Params + PolyMatrix
core** (NOT reusing C++ Spiral's compile-time types).

## Module sizes (spiral-rs)

| module | LOC | port? |
|---|---:|---|
| params.rs | 306 | yes (struct + init; **drop ntt_tables**, see strategy) |
| poly.rs | 1013 | yes (PolyMatrixRaw/NTT + ops; scalar paths only) |
| arith.rs | 532 | partial (barrett + modular arith helpers actually used) |
| ntt.rs | 895 | **NO — replace with HEXL** (proven in chunk 6) |
| number_theory.rs | 96 | partial (modinv / root-of-unity if needed) |
| aligned_memory.rs | 98 | minimal (a std::vector<uint64_t> wrapper is enough) |
| discrete_gaussian.rs | 206 | **already ported** (chunk 4b) |
| gadget.rs | 96 | yes (t_gsw/t_conv gadget decomposition, used by GSW) |

Net new C++ ≈ params (~200 after dropping ntt_tables) + poly scalar ops
(~500–700) + arith barrett (~150) + gadget (~100) ≈ **1000–1200 LOC**,
dense crypto, phased.

## Key strategy (what makes this tractable)

1. **Offload NTT to HEXL.** Do not port ntt.rs (895 LOC) or `Params`'s
   `ntt_tables`. Instead `Params` stores two `intel::hexl::NTT(poly_len,
   moduli[m])` (exactly the chunk-6 pattern). `PolyMatrix::ntt()/raw()`
   call HEXL per modulus + CRT. NB: HEXL's NTT-domain ordering differs
   from spiral_rs — fine as long as **all** NTT-domain ops go through
   the ported PolyMatrix (self-consistent); never mix with spiral_rs
   byte layouts.
2. **Scalar arithmetic only (no AVX512).** `.50` is Broadwell. Port the
   non-AVX paths (`multiply_poly`, `add_poly`, `multiply`, `automorph`,
   …) and skip `*_avx`/`fast_*_avx512`. Perf is a separate concern
   (benchmarks need AVX512 hardware or CUDA, task 8/11).
3. **Reuse what's ported:** discrete_gaussian (4b), negacyclic/util
   (6a/13a), bits (5a), the convolution CRT-NTT pattern (6), ChaCha (2b-iii).
4. **Port the MINIMAL subset incrementally.** Don't port all of poly.rs
   up front — port `Params` + PolyMatrix containers + `ntt`/`raw` +
   `add`/`multiply` first, then add ops (`automorph`, gadget, `*_no_reduce`,
   `add_into_at`) when the consuming chunk (packing/server) first needs them.

## Phased chunks

- **P0 — arith + barrett (~150 LOC).** Port the modular-reduction helpers
  from arith.rs that `Params` and `multiply` need: `barrett_*` constants
  computation (Params::init derives `barrett_cr_0/1[m]`), `barrett_coeff`,
  `barrett_reduction_u128`, `multiply_modular`. Oracle: hand-computed mod
  reductions; cross-check `barrett_coeff(p, x, m) == x % moduli[m]`.
- **P1 — Params struct + init (~200 LOC).** Fields: poly_len, poly_len_log2,
  crt_count, moduli[2], modulus (=product), modulus_log2, n, pt_modulus,
  t_conv/t_exp_left/t_exp_right/t_gsw, expand_queries, instances,
  db_item_size, version, noise_width, barrett constants, **+ two hexl::NTT**.
  Drop `ntt_tables`/`scratch`. Add a `FromJson` (serde→struct) +
  `internal_params_for(...)`. Oracle: build the wiki/test config, assert
  field values (modulus = q0*q1, poly_len, crt_count=2, …) against the
  upstream Rust test params (get_test_params).
- **P2 — PolyMatrix containers + ntt/raw (~250 LOC).** `PolyMatrixRaw`
  (rows×cols polys, flat u64 data, coeff form) + `PolyMatrixNTT`
  (crt_count×poly_len per poly). `ntt()`/`raw()` via the Params' HEXL NTTs
  + CRT (reuse chunk-6 logic). Oracle: `raw(ntt(m)) == m` round-trip;
  cross-check a single-poly convolution against chunk-6 Convolution.
- **P3 — scalar poly arithmetic (~250 LOC).** `add`/`add_poly`,
  `multiply` (NTT-domain pointwise via barrett), `multiply_no_reduce`,
  `add_into`/`add_into_at`, `scalar_multiply`. Oracle: NTT-domain multiply
  then raw == NaiveNegacyclicConvolve of the raw operands.
- **P4 — automorphism + gadget (~250 LOC).** `automorph_poly` (Galois
  X→X^t permutation, used by query expansion) + gadget.rs
  (decompose/`build_gadget` for t_gsw/t_conv, used by GSW). Oracle:
  automorph hand-permutation on small poly_len; gadget·gadget^{-1} identity.
- **P5 — params.rs port complete (~80 LOC).** `ext_params_from_json` +
  `internal_params_for` exposed as the YPIR params entry. Verifies the
  whole stack composes. This unblocks chunk 7 (params), then 9–12.

Each phase: fetch upstream .rs → port scalar path → test vs oracle →
build (`bazel test --config=linux_x86_64`) → commit → push, exactly like
chunks 4–6.

## Risks / open questions

- **barrett constant derivation** must match spiral_rs bit-for-bit or
  NTT-domain multiply diverges — port `Params::init`'s barrett setup
  carefully; test `multiply` against the naive convolution oracle.
- **automorph index math** (`automorph_poly_uncrtd`) is fiddly (negacyclic
  sign flips on the permutation) — hand-verify on poly_len=4/8.
- **HEXL NTT ordering vs spiral_rs:** safe *only* if no ported op ever
  consumes a spiral_rs-serialized NTT-domain buffer. Keep all NTT-domain
  data inside the ported PolyMatrix.
- **AVX512 gap:** kernel.rs (`fast_batched_dot_product_avx512`) and the
  `*_avx` poly ops are unported here; YPIR's hot matmul path will use the
  scalar fallback on `.50` (correct but slow). Real perf needs AVX512
  hardware or the CUDA path (task 8). This port is for *correctness*.
- **Scope creep:** poly.rs has ~30 free functions; port the subset that
  packing/server/scheme/client actually call, incrementally, not all.

## What this unblocks

P0–P5 → chunk 7 (params) → then packing(9)/server(10)/scheme(11)/
client(12) become ports against a real C++ runtime Params + PolyMatrix
(still large, but no longer blocked). The HEXL facade (done) + this
Params/PolyMatrix core are the two foundations the whole upper half needs.
