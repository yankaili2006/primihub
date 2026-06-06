# PIR Benchmark

This document describes the benchmark scripts in `bench/`, captures the
current state of measured-vs-claimed performance, and explains how to
interpret the JSON outputs that the scripts emit.

> **State of the world (2026-06-06):** 1 of 6 registered algorithms is
> real (id_pir / SealPIR). The other 5 are skeletons whose OnExecute
> returns FAIL by design. The bench scripts are structured to give
> useful output *today* (selector regression) and pick up meaningful
> per-algorithm numbers as the real crypto kernels land in
> tasks 4.4 / 5.5 / 7.x.

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
| `double_pir` | 1e8   | ~10 ms          | TBD          | TBD          | (skeleton) |
| `simple_pir` | 1e7   | sub-second      | TBD          | TBD          | (skeleton) |
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
