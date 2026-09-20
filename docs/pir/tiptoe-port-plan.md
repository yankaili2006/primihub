# Tiptoe PIR — C++ port plan (openspec change `primihub-pir-cuda-tiptoe`, task 1.1)

Tiptoe (Henzinger et al., **SOSP'23**) — private nearest-neighbor / semantic
search built as **BFV-on-SimplePIR**. Upstream: `ahenzinger/tiptoe@ba7cae43`
(pinned as `@tiptoe`, BUILD wrapper doc-only since commit `08d44a70`).

## Scope reality check

Upstream is **not** "600 lines" and **not** "100% opaque Go". It is three layers,
and the crypto core is already C++:

| layer | upstream | LOC | port action |
|---|---|---|---|
| **rlwe** (BFV/LHE over SEAL) | `ahenzinger/underhood` `rlwe/rlwe.{cpp,hpp,h}` | ~336 C++ | **reuse/vendor** — already C++ `extern "C"` over SEAL |
| **underhood** (LHE protocol) | `ahenzinger/underhood` `underhood/*.go` | ~470 Go | **port to C++** (thin layer over rlwe + SimplePIR) |
| **search** (app: embeddings/clustering/coordinator) | `tiptoe/search/*.go` | 28 files | **DEFER** (above the PirOperator abstraction) |

The `rlwe.h` surface is a clean C API over Microsoft SEAL:
`context_*` (BFV params), `plaintext_*` / `ciphertext_*` ops, NTT,
`ciphertext_multiply_plain`, `ciphertext_add`,
**`ciphertext_set_inner_product`** (the homomorphic dot product Tiptoe needs),
`key_encrypt` / `key_decrypt` / `key_encrypt_squished`, and serialization. The
Go `rlwe.go` is just cgo bindings over this — primihub does not need the Go.

**primihub already has the two hard dependencies:**
- a ported **SimplePIR core** — `pir::core::Database` / `core::Matrix` +
  `SimpleHintGen` (Init/Setup/Squish) + Query/Answer/Recover (task 7.2). We reuse
  this instead of porting the `henrycg/simplepir` fork.
- **Microsoft SEAL** — already linked via APSI / keyword_pir (`microsoft-apsi`),
  so rlwe's BFV layer has a SEAL toolchain to bind to.

So the real v1 surface is ~340 LOC of reusable C++ (rlwe) + ~550 LOC of Go
protocol to port + reuse of primihub SimplePIR — far smaller than the YPIR port
(5,819 LOC). The estimate in commit `08d44a70` ("~2 weeks") predates the
"rlwe-is-already-C++ + reuse primihub SimplePIR" findings and is conservative.

## v1 scope (decided)

**Core LHE-on-SimplePIR operator.** `TiptoePirOperator::OnExecute` exposes the
Tiptoe cryptographic retrieval — BFV-on-SimplePIR with a homomorphic
inner-product answer — as a PirOperator (`db_content` + `query_indices` →
`recovered`). **Deferred** to a later change/phase: the `search/` application
layer (sentence-transformer embeddings, k-means corpus clustering, the 2-round
nearest-cluster→retrieve coordinator). Those sit above the PirOperator contract
and are an application, not the operator.

## Architecture (v1)

```
TiptoePirOperator::OnExecute
  ├─ rlwe (C++/SEAL, vendored)      : BFV context, ct/pt, inner_product, (de)serialize
  ├─ tiptoe LHE layer (ported)     : client (query/recover), server (answer/hint),
  │                                   params, secret  ── from underhood/*.go
  └─ pir::core SimplePIR (reused)  : Database/Matrix, SimpleHintGen Init/Setup/Squish
```

The LHE answer = SimplePIR linear answer (A·s) **plus** a BFV homomorphic
inner-product over the packed rows, decrypted client-side. underhood's `hint.go`
is the offline hint (per-database), matching `caps.hint_per_database=true`.

## Chunk breakdown

- **1.1a** *(this chunk)* — scope reality check + this plan + **registered
  skeleton operator** (`tiptoe_pir/{h,cc,BUILD}`, `kIsSkeleton=true`,
  `caps.is_real=false`, OnExecute returns FAIL with a clear "not yet vendored"
  log), wired into the `:pir` aggregate + a registry/skeleton test. Mirrors the
  original ypir/spiral/frodo skeleton landings.
- **1.1b** — vendor `underhood/rlwe` as a thirdparty bazel target over primihub's
  SEAL (`thirdparty/pir/BUILD.underhood_rlwe` + WORKSPACE pin of
  `ahenzinger/underhood`), `--define=enable_tiptoe_real=1` gate + override-repo
  validation. Smoke test: BFV encrypt→inner_product→decrypt roundtrip.
- **1.1c** — port `underhood/params.go` + `secret.go` → `tiptoe_params.{h,cc}` +
  `tiptoe_secret.{h,cc}` (pure data + key gen; zero-dep, unit-tested).
- **1.1d** — port `underhood/client.go` → `tiptoe_client.{h,cc}` (query gen +
  recover over rlwe + SimplePIR query).
- **1.1e** — port `underhood/server.go` + `hint.go` → `tiptoe_server.{h,cc}`
  (answer = SimplePIR answer + BFV inner-product; per-database hint).
- **1.1f** — wire `TiptoePirOperator::OnExecute` end-to-end (reuse
  `pir::core` SimplePIR for the linear layer), flip `kIsSkeleton=false` /
  `caps.is_real=true`; e2e retrieval test; flip the spec delta's MODIFIED
  requirement scenario green.

## Notes / risks
- **SEAL version compatibility**: rlwe.cpp pins a SEAL API; primihub's SEAL (via
  APSI) must match. Resolve in 1.1b; if APIs drift, thin-shim the few rlwe.cpp
  calls.
- **.50 is Broadwell (no AVX512)** and the APSI/SEAL externals are GFW-blocked on
  `.50` (see memory `reference_primihub-pir-build-env`). The tiptoe operator will
  be gated like keyword_pir (opt-in `microsoft-apsi`/`enable_tiptoe_real`) so the
  default `.50` build stays green; real-mode build/test needs the SEAL toolchain
  available.
- Capability profile (from the canonical `pir-algo-tiptoe` spec):
  `{query_types=Semantic, servers=1, needs_preprocess=true, hint_per_database=true,
  SemiHonest, SubSecond, recommended_max_db_size=1e8, backends={CPU},
  typical_query_comm_bytes=16384, typical_hint_size_bytes=2e9}`.
