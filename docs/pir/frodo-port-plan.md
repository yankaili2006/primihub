# FrodoPIR C++ Port — Plan (task 7.1)

## Scope reality check

Task 7.1 estimates "**~800 LOC Rust→C++ port**" for FrodoPIR.
Upstream `brave-experiments/frodo-pir@15573960` actually contains
**841 LOC across 5 Rust files** — the original estimate is
almost exactly right (within 5%). Compared with YPIR (5,819 LOC),
FrodoPIR is roughly **1/7 the size** and has **no C/C++ surface**
at all (no `matmul.cpp` shortcut).

Per-file inventory (`wc -l` of `/tmp/frodo-pir-upstream/src/*.rs`):

| File | LOC | Role |
|---|---:|---|
| `api.rs` | 276 | Shard / Server / Client protocol entry points |
| `db.rs` | 260 | Database structure + LWE matrix-form preprocessing |
| `errors.rs` | 57 | Custom error types — `ErrorUnexpectedInputSize`, `ErrorQueryParamsReused`, `ErrorOverflownAdd` |
| `lib.rs` | 4 | Crate root (`pub mod api / mod db / pub mod errors / mod utils`) |
| `utils.rs` | 244 | LWE constants + matrix helpers + bit format helpers |
| **Total** | **841** | |

## Dependency graph

```
errors.rs (zero deps; just std::error::Error impls)
   ▲
   │
utils::lwe (zero deps — 3 const functions)
utils::format ◄── errors.rs (2 of 7 fns return errors)
   │
   ▼
utils::matrices ◄── rand_core (StdRng / OsRng) + SeedableRng
   │
   ▼
db.rs ◄── utils::{format, matrices}, errors, serde, bincode
   │
   ▼
api.rs ◄── db.rs, utils::{lwe, format, matrices}, errors, serde
```

**No Spiral coupling** — unlike YPIR's blockers, FrodoPIR is
self-contained. The blocker for the algorithmic core is just
**LWE matrix multiplication over u32** (utils::matrices), which
upstream implements naively in Rust. A C++ port can either:
- reuse primihub's existing `pir_core::Matrix` (which already
  handles u32 LWE matrix arithmetic for SimplePIR / DoublePIR), or
- write a fresh `frodo_matrix` companion.

For the first chunk we don't have to decide yet — chunks 1+2
cover constants and bit-level format helpers, both zero-dep.

## Port order (recommended)

| # | Module | Upstream LOC | Spiral dep? | Why this order |
|---|---|---:|:-:|---|
| 1 | `utils::lwe` (3 const fns) | 22 | no | Zero-dep entry chunk, mirrors YPIR transpose.rs choice |
| 2 | `utils::format` (5 of 7 pure fns + 2 error-returning) | 100 | no | Bit-level format helpers; pure subset + 2 helpers swapped to `retcode + err string` |
| 3 | `errors.rs` skip (use existing retcode/err pattern) | — | n/a | Don't port the custom error types; map to retcode + std::string at the wrapper boundary |
| 4 | `utils::matrices` | 100 | partial | Decision point: reuse `pir_core::Matrix` or fresh `frodo_matrix`. Needs a deterministic seedable PRNG to match upstream `StdRng::seed_from_u64`. |
| 5 | `db.rs` Database struct + LWE preprocessing | 260 | — | DB compression + per-row LWE encoding. Bulk of the algorithmic port. |
| 6 | `api.rs` Client/Server protocol | 276 | — | Shard / Server / Client lifecycle. End-to-end on top of 1-5. |
| 7 | OnExecute wiring | ~80 | — | Replace skeleton operator with real Client/Server roundtrip. |
| 8 | E2E test | ~120 | — | Verify a single-row lookup against the upstream test fixture in `data/`. |

## Chunk 1 (this session): `utils::lwe` + `utils::format` (pure subset)

**Rationale**: smallest standalone unit with real semantics. LWE
constants are 3 trivial integer-math functions (rounding factor /
floor / plaintext size); format helpers are 5 pure bit-level
functions (`u8_to_bits_le`, `u32_to_bits_le`, `bits_to_bytes_le`,
`bytes_to_bits_le`, `bytes_from_u32_slice`). The roundtrip
`bytes ↔ bits ↔ bytes` is the natural cornerstone test.

The two error-returning format helpers (`bits_to_u32_le`,
`u32_sized_bytes_from_vec`) are also portable today — they just
need the C++ wrapper to use the existing `retcode + std::string`
pattern instead of the custom error types. Included in chunk 1.

The base64 helper `base64_from_u32_slice` depends on the `base64`
crate; we skip it (downstream tests should not depend on a base64
crate either).

**Deliverables**:
- `src/primihub/kernel/pir/operator/frodo_pir/frodo_lwe_consts.{h,cc}`
- `src/primihub/kernel/pir/operator/frodo_pir/frodo_format.{h,cc}`
- `src/primihub/kernel/pir/tests/frodo_lwe_consts_test.cc`
- `src/primihub/kernel/pir/tests/frodo_format_test.cc`
- BUILD edits in `frodo_pir/BUILD` and `tests/BUILD`
- `bazel test //src/primihub/kernel/pir/tests:frodo_lwe_consts_test
                  //src/primihub/kernel/pir/tests:frodo_format_test`
  PASS in default mode (no @frodo_pir dep needed — we don't link
  against upstream Rust)

**Out of scope for chunk 1**: integration with `FrodoPirOperator`
(stays a skeleton until chunk 7). The new helpers live in their
own cc_libraries so the existing skeleton compiles unchanged.

## Naming convention

Port helpers go in **`primihub::pir::frodo` namespace** (parallel
to `primihub::pir::ypir` from task 7.3). The existing
`FrodoPirOperator` is in `primihub::pir` for compatibility with
the registry's `factory.h` pattern; that stays unchanged.

This means the chunk-1 functions are referenced as
`primihub::pir::frodo::GetRoundingFactor` etc. Tests use a
`using namespace primihub::pir::frodo;` inside the anonymous
namespace to keep call sites readable, just as `ypir_*_test.cc`
does for `primihub::pir::ypir`.
