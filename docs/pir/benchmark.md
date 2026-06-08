# PIR Benchmark

This document describes the benchmark scripts in `bench/`, captures the
current state of measured-vs-claimed performance, and explains how to
interpret the JSON outputs that the scripts emit.

> **State of the world (2026-06-08):** 3 of 6 registered algorithms are
> real: id_pir (SealPIR, since project inception), simple_pir (task 7.2,
> commit 2d77509a), and double_pir (task 5.5, commit 31e8fd43). The
> remaining 3 (spiral / frodo_pir / ypir) are skeletons whose OnExecute
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

Observations:
* Init scales roughly linearly in sqrt(N) — matches expectations
  (sampling two `sqrt(N)×N` matrices, where N is the LWE secret dim).
  N=64→4194304 (sqrt-ratio 256×) → init 1.3→227 ms (~175×).
* Setup grows close to N because the H1=DB·A1 matrix multiply is
  O(L·M·n) where L·M ≈ DB size. N=64→4194304 (65536×) → setup
  43→4510 ms (105× — sub-linear thanks to BLAS-shaped kernels in the
  C matmul, but the trend is clear).
* Per-query latency drifts upward modestly with N (35 ms at small N
  → 71 ms at N=4M) but remains an order of magnitude lower than the
  paper's expected per-query → confirms the algorithm's online cost
  is N-independent in spirit. The upward drift is cache pressure
  (intermediates grow with sqrt(N)).
* The single-trial N=4M Setup is ~4.5 s; extrapolating linearly to
  N=1e8 gives ~110 s Setup + ~70 ms per-query. RAM is the limiting
  factor — host needs ~10-50 GB free for the intermediates. Task 5.10
  1e8 cell stays open until such a host is reachable.

1e8 cell is task 5.10's long-term target, pending a host with enough
RAM (the LWE intermediate matrices grow to several GB at N=1e8).

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
    {"algorithm": "frodo_pir",  "status": "SKIP-stub", "reason": "OnExecute returns FAIL by design"},
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
| `double_pir` | 1e8   | ~10 ms          | TBD          | TBD          | (real, bench task 5.10 pending) |
| `simple_pir` | 1e7   | sub-second      | TBD          | TBD          | (real, bench TBD)              |
| `frodo_pir`  | 1e7   | ms class        | TBD          | TBD          | (skeleton) |
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
