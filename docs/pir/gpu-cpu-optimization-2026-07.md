# DoublePIR GPU + CPU optimization (2026-07)

A profile-driven pass over the DoublePIR path: GPU-offload where the work is
compute-bound, CPU allocation/copy elimination where it is memory-bound. Every
step was decided by `perf`, and every behavioral change was gated by the
value-asserting end-to-end tests. Validated on an NVIDIA RTX 5070 Ti / CUDA 13.2.

## TL;DR

| Stage | Bottleneck (measured) | Change | Result |
|-------|-----------------------|--------|--------|
| Setup | `db * A1`, `H1 * A2` matmuls (compute) | `Matrix::Mul`/`MulVec` -> CUDA `LweMatMulMod2Pow32` | **up to 7.7x @ N=16.7M** (6574 -> 855 ms) |
| Answer (per-query) | *not* the matvec | packed CUDA `PackedMatVecMod2Pow32` wired to `MulVecPacked` | correct end-to-end; parity-only (see below) |
| Recover (per-query) | Matrix alloc / zero-fill / copy (memory) | `SelectRows` range-assign; inline `Get`/`Set`; pre-sized `VConcatRows` | **per-query up to ~3.6x** (40 -> 11 ms @ N=262144) |

## Method

`perf record -g --call-graph dwarf` on `double_pir_latency_bench` sized so the
target stage dominates wall time (e.g. small N + many queries to isolate the
per-query path). Read the flat (self) and inclusive (children) views; fix the
top actionable cost; re-profile; stop when the remaining cost is inherent.

## GPU: Setup matmul (compute-bound -> real win)

`Matrix` is a row-major uint32 matrix (mod 2^32) shared by SimplePIR / DoublePIR
/ YPIR, and its `Mul`/`MulVec` bridge to the `@simplepir` C kernels is the
DoublePIR Setup hot path (`db.Mul(A1)`, `H1.Mul(A2)`). Under `PIR_CORE_CUDA`
(set by `--define=enable_cuda=1`) these dispatch to the validated
`double_pir/cuda` kernel `LweMatMulMod2Pow32` (`C = A*B mod 2^32`, tiled matmul +
warp-per-row matvec). Env `PIR_CORE_CUDA_FORCE=1` / `PIR_CORE_CUDA_DISABLE=1`
force the path; otherwise it triggers above ~4M MACs so tiny matmuls stay on CPU.

Because `matMul` is shared across all three algorithms, this one hook is the
highest-leverage GPU integration point.

Setup latency, CPU vs GPU:

| N | CPU setup | GPU setup | speedup |
|---|-----------|-----------|---------|
| 262,144 | 729 ms | 251 ms | 2.9x |
| 1,048,576 | 1320 ms | 269 ms | 4.9x |
| 4,194,304 | 2533 ms | 439 ms | 5.8x |
| 16,777,216 | 6574 ms | 855 ms | **7.7x** |

Speedup grows monotonically with N (a CPU-only path would be flat), and the
GPU-forced run bumps device memory + shows nonzero SM utilization -- positive
confirmation the CUDA path executes.

## Answer: packed matvec kernel (correct, but not the bottleneck)

`PackedMatVecMod2Pow32` mirrors the AVX2 `matMulVecPacked` for the squished DB
(basis=10, squishing=3): `out[i] = sum_j sum_{s<3} ((a[i*cols+j]>>10s)&1023) *
b[3j+s] mod 2^32`, warp-per-row + shuffle-reduce. Wired to `Matrix::MulVecPacked`
and validated (standalone 0-mismatch; forced-GPU end-to-end recovery correct).

**But per-query latency did not improve.** The Answer matvec reads a ~22 MB
squished DB (N=16.7M): ~37 us on the GPU at 600 GB/s, vs ~100 ms total
per-query -- under 1% of the cost. Even with a device-resident DB cache (env
`PIR_CORE_CUDA_RESIDENT=1`, removing per-call H2D) per-query was unchanged. The
per-query bottleneck is elsewhere (Recover, below), so the packed kernel is
kept as correct infrastructure, not a win by itself.

## Recover: CPU allocation churn (memory-bound -> real win)

`perf` showed per-query was ~77% `Recover`, dominated NOT by compute (all
matmuls ~12%) but by `Matrix` memory churn: ctor zero-fill 33%, `SelectRows`
38%, `Concat` 19%, `Get` 13%. Three changes, all behavior-preserving:

1. **`SelectRows`**: build the row block with a `std::vector` range-assign
   instead of `Matrix(n, cols)` (zero-fill) + `std::copy` (overwrite). The
   zero-fill was pure waste -- every element was immediately overwritten.
2. **`Get`/`Set`**: moved out-of-line definitions into the header (inline),
   removing a function call per element access across a TU boundary.
3. **`VConcatRows(a, offA, nA, b, offB, nB)`**: stacks two row ranges into one
   matrix via `reserve` + two `insert`s -- a single allocation, no zero-fill,
   no realloc. Replaces the `SelectRows()+Concat()` pairs in Recover's inner
   loop (which allocated, then reallocated-and-moved on the append).

Per-query latency (CPU), cumulative:

| N | before | after | speedup |
|---|--------|-------|---------|
| 262,144 | ~40 ms | 11 ms | ~3.6x |
| 1,048,576 | ~41 ms | 19 ms | ~2.1x |
| 4,194,304 | ~59 ms | 37 ms | ~1.6x |
| 16,777,216 | ~101 ms | 81 ms | ~1.25x (compute-bound at large N) |

The win is largest at small N where memory churn dominated; at large N the
actual matmul/compute grows and caps the relative gain.

### Where per-query stops

A follow-up profile put `Matrix::Get` at 24% self-time. Splitting its cold
`LOG(FATAL)` path out-of-line to force inlining changed nothing -- confirming
that 24% is the *memory-access latency* of Recover's element loops (perf
attributes the load to `Get`), not call overhead. Per-query is memory-bound at
these sizes; further gains need algorithmic vectorization / cache-layout work.

## Correctness

Gated by the value-asserting end-to-end tests -- `double_pir_protocol_test`,
`double_pir_test`, `double_pir_role_test` (full Query/Answer/Recover with value
checks) -- all PASS after every change. Note: `double_pir_runtime_test`
(SmokeIsIdempotent) only checks success + idempotency, *not* recovered values,
so it is **not** a sufficient gate for Recover changes; use the three above.

## Build

```
# CPU
bazel build --config=linux_x86_64 \
  --define=enable_pir_core_real=1 --define=enable_double_pir_real=1 \
  //src/primihub/kernel/pir/bench:double_pir_latency_bench

# GPU (adds the nvcc genrule, -arch=sm_120)
bazel build ... --define=enable_cuda=1 \
  --action_env=PATH=/usr/local/cuda/bin:/usr/bin:/bin
```

## Follow-ups (diminishing returns; separate work)

- **Per-query GPU**: needs a device-resident DB + per-matrix LRU (the Answer
  alternates two matrices); naive per-call H2D is PCIe-bound and does not beat
  CPU. Only worth it once per-query is not Recover-bound.
- **Recover memory floor**: vectorize the element loops / improve cache layout;
  or pool buffers to cut first-touch page faults.
