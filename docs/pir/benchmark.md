# PIR Benchmark

This document describes the benchmark scripts in `bench/`, captures the
current state of measured-vs-claimed performance, and explains how to
interpret the JSON outputs that the scripts emit.

> **State of the world (2026-06-08):** 3 of 6 registered algorithms are
> real: id_pir (SealPIR, since project inception), simple_pir (task 7.2,
> commit 2d77509a), and double_pir (task 5.5, commit 31e8fd43). The
> remaining 2 (spiral / ypir) are skeletons whose OnExecute
> returns FAIL by design — real crypto kernels land in tasks 4.4 / 7.1
> / 7.3. The bench scripts are structured to give useful output *today*
> (selector regression + correctness smoke for the real ones) and pick
> up per-algorithm latency numbers as the remaining ports land.

---

## Scripts

### `bench/pir_selector_sweep.sh`

Walks the selector input space — `db_size × query_type × latency_budget
× allow_two_server × client_can_cache_hint × assume_non_colluding` —
and captures which algorithm wins each cell, with its score and the
failure reasons for the algorithms that lost.

What this catches:

- **Selector regressions.** If a refactor accidentally makes DoublePIR
  lose for `db_size=1e8, latency=ms, allow_two_server=true,
  assume_non_colluding=true, client_can_cache_hint=true`, diffing the
  sweep JSON catches it immediately. The cells are stable across
  builds when the capability data hasn't changed.
- **New-algo integration smoke.** When a new algorithm registers, its
  rows show up in the per-cell rankings; cells where it newly wins are
  visible as `winner` changes.

What this doesn't catch:

- **Real crypto correctness.** The selector consults capability
  metadata, not the underlying operator's OnExecute. A skeleton
  algorithm still gets ranked normally.
- **Actual latency.** "Score" is a selector-internal heuristic; it
  does not reflect microsecond timings.

Usage:

```bash
# From inside the primihub worktree, after a build:
bazel build --config=linux_x86_64 //src/primihub/cli/pir_inspect:pir_inspect
bench/pir_selector_sweep.sh
# → bench/results/pir_selector_sweep_<YYYYMMDD>.json

# Compare against last known-good sweep:
diff -u bench/results/pir_selector_sweep_<old>.json \
        bench/results/pir_selector_sweep_<new>.json
```

### `bench/pir_correctness_smoke.sh`

For each registered algorithm, classifies it as PASS / SKIP-stub / SKIP
based on whether OnExecute is real or a skeleton. The script's value is
in **maintaining the invariant** that a skeleton stays a skeleton until
someone consciously promotes it.

CI gate idea: fail the build if a previously-skeleton algorithm now
reports PASS without a matching `tasks.md` update marking the real
crypto kernel as landed.

Usage:

```bash
bench/pir_correctness_smoke.sh
# → bench/results/pir_correctness_smoke_<YYYYMMDD>.json

# Opt into actually running primihub-cli for the real algos:
PIR_SMOKE_RUN_TASK=1 bench/pir_correctness_smoke.sh
```

---

### `bench/double_pir_latency_bench.sh`

DoublePIR latency sweep across a range of DB sizes. Drives the cc_binary
`src/primihub/kernel/pir/bench/double_pir_latency_bench` once per N
value, captures setup + per-query latency, and emits JSON matching the
other bench scripts' schema_version=1 shape.

Two-call measurement: one OnExecute with 1 query (cost = Setup + per-query),
one with --queries=Q (cost = Setup + Q · per-query). Subtraction gives
per-query latency. --trials averages multiple repetitions per N.

Gated behind `PIR_BENCH_RUN_BAZEL=1` + `SIMPLEPIR_UPSTREAM=<path>` so a
fresh checkout doesn't trigger a 30 s build on every CI invocation.
Usage:

```bash
PIR_BENCH_RUN_BAZEL=1 SIMPLEPIR_UPSTREAM=/tmp/simplepir-upstream \
    bench/double_pir_latency_bench.sh --trials 3
```

#### Baseline on .50 (2026-06-08, schema_version=2, 3 trials, 16 queries each)

Per-stage timings parsed from the operator's structured timing LOG (added
in commit ee01f2*) — `init_ms` is the cost of sampling public matrices
A1+A2; `setup_ms` is the per-database H1/H2/A2_copy preprocessing;
`per_query_ms` is the average of the Query+Answer+Recover loop body.

| N        | sqrt(N) | init_ms | setup_ms | per_query_ms | trials |
|----------|---------|---------|----------|--------------|--------|
| 64       | 8       | 1.3     | 42.5     | 43.3         | 3      |
| 256      | 16      | 2.5     | 52.5     | 32.9         | 3      |
| 1024     | 32      | 4.8     | 70.4     | 28.3         | 3      |
| 4096     | 64      | 9.6     | 121.8    | 34.5         | 3      |
| 16384    | 128     | 18.8    | 354.1    | 43.0         | 3      |
| 65536    | 256     | 28.1    | 447.9    | 34.8         | 3      |
| 262144   | 512     | 110.3   | 1536.2   | 54.0         | 2      |
| 1048576  | 1024    | 134.3   | 2093.0   | 50.5         | 2      |
| 4194304  | 2048    | 226.6   | 4509.9   | 71.3         | 2      |
| 1000000  | 1000    | 95.4    | 1714.4   | 53.1         | 1      |
| 4000000  | 2000    | 193.5   | 5414.3   | 81.3         | 1      |
| 16000000 | 4000    | 447.8   | 12307.0  | 136.2        | 1      |
| 64000000 | 8000    | 1101.9  | 36367.5  | 273.5        | 1      |
| **100000000** | **10000** | **1004.7** | **53941.3** | **335.1** | **1** |

Note: the 1e8 row was added 2026-06-10 by the task 5.10 large-scale sweep
(see `bench/results/double_pir_latency_largescale_20260610T024909Z.json` —
binary sha256 `e3e5776f9aadb8aa9203f4eca4730e26ef7bffb7e1d6c1d47fc71c2f8c85bf0c`).
Constraint: `--n` must be a perfect square with `sqrt(N) % 8 == 0`
(DoublePirOperator's `kRowAlignment` guard); valid large-N points are
1e6, 4e6, 16e6, 64e6, 1e8.

Observations:
* Init scales roughly linearly in sqrt(N) — matches expectations
  (sampling two `sqrt(N)×N` matrices, where N is the LWE secret dim).
  N=64→1e8 (sqrt-ratio 1250×) → init 1.3→1005 ms (~770× — sub-linear
  thanks to BLAS-shaped kernels).
* Setup grows close to N because the H1=DB·A1 matrix multiply is
  O(L·M·n) where L·M ≈ DB size. N=64→1e8 (1.56M×) → setup
  43→53941 ms (1250×). Sub-linear scaling holds across 6 decades.
* Per-query latency grows with N (43 ms at small N → 335 ms at N=1e8).
  Online cost is dominated by the LWE decoder's matrix accesses — at 1e8
  the squished DB chunk hit is ~256 MB which exceeds L3, so cache misses
  dominate. The upstream Go reference reports ~17 ms per-query at 1e8 on
  a tuned server (AVX-512 fp32 kernels); our pure C bridge is ~20× slower
  but functionally correct.
* RAM at N=1e8: ~10-15 GB peak (DB matrix 400 MB + A1/A2 hint
  intermediates 1-2 GB + Setup expansion scratch). .50 at 32 GB total
  reached ~7 GB free during the run, no OOM.

Task 5.10 1e8 cell **landed 2026-06-10** on .50; this is the first
real-data point at the algorithm's design target scale in primihub.

### `bench/double_pir_persistence_bench.sh` (task 5.6 chunks 1-5)

Runs each N twice against the same hint file — first cold (rm -f
the file first), second warm — then emits JSON with cold + warm
sub-objects and a `wall_speedup` ratio. The bench cc_binary
(`double_pir_latency_bench --hint-path PATH`) drives
`DoublePirOperator` through `MaybeLoadOnce` + `SaveToFile`; on the
warm pass the cache hit short-circuits the O(L·M·n) Setup.

Baseline on .50 (queries=4):

| N    | cold setup_ms | warm setup_ms | wall speedup |
|------|---------------|---------------|--------------|
| 64   | 43            | 0             | 1.6×         |
| 256  | 60            | 0             | 1.7×         |
| 1024 | 97            | 0             | 1.8×         |
| 4096 | 224           | 0             | 3.4×         |

The speedup grows with N because cold Setup is O(L·M·n) while warm
Setup is fixed at 0. The 2026-06-10 large-scale sweep (above) measured
cold Setup=53.9 s at N=1e8 vs the upstream paper's ~10 ms per-query;
a warm hit would give a wall speedup of ~150× at that scale (53.9 s
vs ~335 ms per-query).

### `bench/simple_pir_persistence_bench.sh` (SimplePIR sibling)

Same shape as the DoublePIR wrapper but drives
`simple_pir_latency_bench` → `SimplePirOperator`. SimplePIR's hint
is just `{A, H}` (vs DoublePIR's 5-matrix bundle), and the online
path is one `MulVecPacked` + `Recover`, so absolute wall times are
1-2 orders of magnitude lower than DoublePIR's at the same N.

Baseline on .50 (queries=4):

| N        | cold (Init+Setup)_ms | warm (Init+Setup)_ms | per_query_ms | wall speedup |
|----------|----------------------|----------------------|--------------|--------------|
| 64       | 0.7                  | 0                    | n/a          | 2.3×         |
| 256      | 1.5                  | 0                    | n/a          | 2.5×         |
| 1024     | 3.3                  | 0                    | n/a          | 2.7×         |
| 4096     | 7.5                  | 0                    | n/a          | 2.7×         |
| 16384    | 19                   | 0                    | n/a          | 2.4×         |
| 65536    | 55                   | 0                    | n/a          | 4.1×         |
| 262144   | 145                  | 0                    | n/a          | 4.1×         |
| 1048576  | 381                  | 0                    | n/a          | 3.0×         |
| 4194304  | 2205                 | 0                    | n/a          | 3.7×         |
| 16000000 | 6348                 | 0                    | 14.2 → 13.5  | 3.0×         |
| 64000000 | 22597                | 0                    | 36.4 → 36.4  | 3.2×         |
| **100000000** | **38413**       | **0**                | **51.8 → 49.5** | **3.2×** |

Note: the 16M / 64M / 1e8 rows landed 2026-06-10 alongside the DoublePIR
1e8 measurement (task 5.10) — same sweep methodology, same .50 host.
SimplePIR's per-query at 1e8 is **6.5× faster than DoublePIR's at the same
scale** (52 ms vs 335 ms), reflecting SimplePIR's single matrix
multiplication online cost vs DoublePIR's nested LWE recovery. Result
JSON: `bench/results/simple_pir_persistence_largescale_20260610T030543Z.json`,
binary sha256 `94a706ff8e04eed5064bedfd0f897e8b19c630b612aafbef4e711fe5bd1b1220`.

Peak speedup at moderate N (~4×); at very large N the cache-file
load + per-query work start to claim a non-trivial fraction of the
warm wall, capping the ratio. Setup work itself stays fully
short-circuited (warm `init_ms = setup_ms = squish_ms = 0`).

Reproduce via `bench/simple_pir_persistence_bench.sh --n-list '...'`.

## Result file shapes

### `pir_selector_sweep_*.json`

```json
{
  "schema_version": 1,
  "captured_at": "2026-06-06T00:00:00Z",
  "binary_path": "bazel-out/.../pir_inspect",
  "binary_sha256": "...",
  "cells": [
    {
      "constraints": {
        "db_size": 100000000,
        "query_type": "index",
        "latency_budget": "ms",
        "allow_two_server": true,
        "client_can_cache_hint": true,
        "assume_non_colluding": true
      },
      "winner": "double_pir",
      "ranking": [
        {"algorithm": "double_pir", "passes": true,  "score": 1000980, "comm_kb": 4,   "fail_reasons": ""},
        {"algorithm": "frodo_pir",  "passes": true,  "score": 999896,  "comm_kb": 64,  "fail_reasons": ""},
        {"algorithm": "id_pir",     "passes": false, "score": 0,       "comm_kb": 0,   "fail_reasons": "db_size far exceeds recommended_max_db_size"}
      ]
    }
  ]
}
```

### `pir_correctness_smoke_*.json`

```json
{
  "schema_version": 1,
  "captured_at": "2026-06-06T00:00:00Z",
  "binary_path": "bazel-out/.../pir_inspect",
  "binary_sha256": "...",
  "results": [
    {"algorithm": "id_pir",     "status": "SKIP",      "reason": "set PIR_SMOKE_RUN_TASK=1 to attempt real cli run"},
    {"algorithm": "apsi",       "status": "SKIP",      "reason": "set PIR_SMOKE_RUN_TASK=1 to attempt real cli run"},
    {"algorithm": "spiral",     "status": "SKIP-stub", "reason": "OnExecute returns FAIL by design"},
    {"algorithm": "double_pir", "status": "SKIP-stub", "reason": "OnExecute returns FAIL by design"},
    {"algorithm": "simple_pir", "status": "SKIP-stub", "reason": "OnExecute returns FAIL by design"},
    {"algorithm": "frodo_pir",  "status": "PASS",      "reason": ""},
    {"algorithm": "ypir",       "status": "SKIP-stub", "reason": "OnExecute returns FAIL by design"}
  ]
}
```

---

## Claimed vs measured (placeholder)

This table gets filled in as the real crypto kernels land. For now it
captures **claimed** numbers from each paper for sizing reference. Each
row will be replaced by a measured tuple `(p50, p95)` once the
corresponding bench/<algo>_e2e.sh script lands per task 4.8 / 5.10.

| algorithm    | scale | claimed latency | measured p50 | measured p95 | binary sha |
|--------------|-------|-----------------|--------------|--------------|------------|
| `id_pir`     | 1e6   | 1-3 s           | TBD          | TBD          | TBD        |
| `apsi`       | 1e6   | sub-second      | TBD          | TBD          | TBD        |
| `spiral`     | 1e8   | 2-3 s           | TBD          | TBD          | (skeleton) |
| `double_pir` | 4M    | (claim is 1e8)  | per-query ~71ms | TBD       | bench/double_pir_latency_bench.sh (.50) |
| `double_pir` | 1e8   | ~10 ms          | per-query 335 ms (single trial) | TBD | e3e5776f9aadb8aa9203f4eca4730e26ef7bffb7e1d6c1d47fc71c2f8c85bf0c (task 5.10 landed 2026-06-10 on .50) |
| `simple_pir` | 4M    | sub-second      | per-query ~8ms  | TBD       | bench/simple_pir_persistence_bench.sh (.50) |
| `simple_pir` | 1e7   | sub-second      | TBD          | TBD          | (real, persistence bench peaks at ~4× speedup) |
| `simple_pir` | 1e8   | sub-second      | per-query 52 ms (single trial) | TBD | 94a706ff8e04eed5064bedfd0f897e8b19c630b612aafbef4e711fe5bd1b1220 (landed 2026-06-10 alongside DoublePIR 1e8) |
| `frodo_pir`  | 1e6   | ms class (paper) | per-query 2308 ms (single trial) | TBD | 00b0b9ae6e3736ba45868141290df3e2ead566e3ee5dd4743eda2f5806648150 (task 7.1 landed 2026-06-12 on .50, unoptimised C++ port) |
| `frodo_pir`  | 1e7   | ms class (paper) | TBD (port unoptimised, single-trial >12 min wall on .50; revisit after SIMD) | TBD | (skipped, see 1e6 baseline) |
| `ypir`       | 1e8   | sub-second      | TBD          | TBD          | (skeleton) |

`pir_matrix_bench.sh` (planned as task 10.1) will be the matrix runner
that automates this table. It will pull from the same source-of-truth
JSON the sweep + correctness scripts emit, so the trio composes into a
single CI-runnable bundle.

---

## Reading the score field

The selector's `score` is a single uint64 that bakes together:

- Base score (1,000,000)
- Minus `PerfClassRank × 100,000` (worse perf class subtracts more)
- Minus typical-comm-bytes penalty (bandwidth_priority mode scales this up)
- Plus `ThreatModelRank × 1,000` (stronger threat model preferred)
- Minus a 50 GB / 20,000-point penalty for very large per-DB hints
- Penalties accrue when `recommended_max_db_size` is exceeded

The intent is "higher score = better at this constraint set". The
absolute number is meaningless; only differences between algorithms in
the same cell matter. Use the sweep JSON to spot when relative ordering
changes.

---

## See also

- [multi-algo-guide.md](multi-algo-guide.md)
- [threat-model.md](threat-model.md)
- [hint-lifecycle.md](hint-lifecycle.md)
- `src/primihub/kernel/pir/operator/selector.cc` — `Evaluate()` is where
  the score formula lives. When the formula changes, rerun the sweep
  and commit the new JSON.
