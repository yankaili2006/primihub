# YPIR C++ Port — Plan (task 7.3)

## Scope reality check

`openspec/changes/primihub-pir-multi-algo/tasks.md` 7.3 currently
estimates "**~600 LOC Rust→C++ port**" for the YPIR algorithmic layer.
Upstream `menonsamir/ypir@a73e550a` actually contains **5,819 lines of
Rust** across 16 files. Even the subset cited in `thirdparty/pir/BUILD.ypir`
("lwe.rs ~150 LOC, packing.rs ~250 LOC, manageable port") underestimates
`packing.rs` by 6×: actual is **1,496 LOC**.

The faithful per-file inventory (`wc -l` of `/tmp/ypir-upstream/src/*.rs`
on .50):

| File | LOC | Role |
|---|---:|---|
| `bits.rs` | 153 | Bit-level packing helpers |
| `client.rs` | 390 | Client setup / query / decode |
| `convolution.rs` | 282 | Polynomial convolution support |
| `kernel.rs` | 189 | SIMD batched dot product wrappers |
| `lib.rs` | 17 | Crate root |
| `lwe.rs` | 128 | LWE primitive (sample, encode, decode) |
| `matmul.rs` | 155 | Rust shim around `src/matmul.cpp` kernels |
| `measurement.rs` | 62 | Bench instrumentation (skippable) |
| `modulus_switch.rs` | 63 | Q-to-q rounding |
| `noise_analysis.rs` | 313 | Param-fitness predicates (skippable for port) |
| `packing.rs` | **1,496** | Hint generation + result packing |
| `params.rs` | 179 | Parameter struct ("YpirParams" facade over `spiral_rs::Params`) |
| `scheme.rs` | 722 | High-level Setup / Query / Answer / Recover orchestration |
| `server.rs` | **1,246** | Server-side hint + answer pipeline |
| `transpose.rs` | 107 | Plain byte / element transpose, tiled f64 |
| `util.rs` | 317 | Generic helpers (skippable selectively) |
| **Total** | **5,819** | |

Realistic estimate: porting the **load-bearing 4,500 LOC** (skip
`measurement`, `noise_analysis`, part of `util`) is a **multi-week
engagement**, not a single sprint. Task 7.3 should be split into the
chunks below; finishing one chunk per session is realistic.

## Dependency graph

```
matmul.cpp (already wrapped via @ypir//:ypir_matmul_kernels) ◄── matmul.rs
                                                                    │
transpose.rs (zero deps) ◄────────────── kernel.rs ◄─────── server.rs
                                  │                          │
                                  └─── bits.rs ◄─── lwe.rs   │
                                            (spiral_rs)      ▼
                                                          packing.rs
                                                              │
                                  params.rs (spiral_rs) ◄─────┘
                                              │
                                              ▼
                                          scheme.rs
                                              │
                                              ▼
                                          client.rs
```

**Spiral coupling**: `params.rs`, `kernel.rs`, `bits.rs`, and parts of
`packing.rs` depend on `spiral_rs` types (`Params`, `AlignedMemory64`,
`arith::*`). Spiral has its own C++ port in progress as task 3 / Phase 3,
which is **partial** per `primihub_pir_spiral_build_state.md`. YPIR
modules that need Spiral types must wait for those concrete C++ types
to land first.

## Port order (recommended)

| # | Module | LOC | Spiral dep? | Why this order |
|---|---|---:|:-:|---|
| 1 | `transpose.rs` | 107 | no | Zero-dep entry chunk; validates per-file pattern |
| 2 | `lwe.rs` | 128 | partial (own LWEParams) | Self-contained LWE primitive |
| 3 | `bits.rs` | 153 | yes (AlignedMemory64) | Wait for SpiralAlignedMem facade |
| 4 | `modulus_switch.rs` | 63 | minor | Numeric helpers |
| 5 | `kernel.rs` | 189 | yes | Wraps matmul kernels; needs Spiral arith |
| 6 | `convolution.rs` | 282 | partial | Polynomial helpers |
| 7 | `params.rs` | 179 | yes | YpirParams struct (after Spiral Params) |
| 8 | `matmul.rs` | 155 | no | Rust shim over already-wrapped C++ kernels |
| 9 | `packing.rs` | 1,496 | yes | Largest; depends on bits/kernel/params |
| 10 | `server.rs` | 1,246 | yes | Server-side Setup / Answer using packing |
| 11 | `scheme.rs` | 722 | yes | Orchestration |
| 12 | `client.rs` | 390 | yes | Client-side Query / Recover |
| 13 | OnExecute wiring | ~100 | — | Replace the smoke-only OnExecute |
| 14 | E2E test | ~150 | — | 64-query batch correctness like SimplePIR |

**Skipped** (not on critical path):
- `measurement.rs` (62) — bench instrumentation, not algorithmic
- `noise_analysis.rs` (313) — param-fitness oracle; can fall back to
  using upstream defaults (`scheme::params_for_data_size_sub_1gb` etc.)
- ~half of `util.rs` (317) — port only what's actually called

## Chunk 1 (this session): `transpose.rs`

**Rationale**: pure byte-shuffle, zero deps, has its own unit test in
the upstream that we can replicate verbatim. Validates the "one file
per chunk" working pattern, gives YPIR a real (non-smoke) test in
the regression suite, and unblocks `kernel.rs` (which uses transposed
B matrices in the dot-product kernel) once that chunk lands.

**Deliverables**:
- `src/primihub/kernel/pir/operator/ypir/ypir_transpose.{h,cc}`
- `src/primihub/kernel/pir/tests/ypir_transpose_test.cc`
- `ypir/BUILD` adds `:ypir_transpose` cc_library
- `tests/BUILD` adds `ypir_transpose_test` cc_test
- `bazel test //src/primihub/kernel/pir/tests:ypir_transpose_test` PASS
  in default (non-vendored) mode (no @ypir dep needed)

**Out of scope for chunk 1**: wiring transpose into `OnExecute`. That
will happen after `scheme.rs` lands (chunk 11) — transpose is a
helper called by `server.rs` packing, not the operator entry point.
