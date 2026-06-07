# PIR Multi-Algorithm Guide

> Status: **6 algorithms registered (1 real + 5 skeletons; 3 of those have C/C++ kernel link activated)** as of 2026-06-07.
> See [benchmark.md](benchmark.md) for the live capability matrix output and
> [threat-model.md](threat-model.md) for per-algo security assumptions.

primihub PIR went from a hard-coded `Factory::Create(PirType, Options)`
switch-case to a multi-algorithm framework where new schemes plug in by
registering with a `PirRegistrar`. This guide is the user-facing handbook
for choosing one.

---

## TL;DR — pick an algorithm

```text
  Q: Are queries by index or by keyword?
      keyword → "apsi" (only keyword PIR; needs --define microsoft-apsi)
      index   → continue ↓

  Q: How big is the database?
      ≤ 1e6 rows         → "id_pir"     (SealPIR, already production)
      1e6 – 1e8 rows     → continue ↓
      ≥ 1e8 rows         → continue ↓

  Q: Can you stand up TWO non-colluding servers?
      yes                → "double_pir"   (10ms at 1e8 — best in class)
      no                 → continue ↓

  Q: Can the client cache a hint blob? (one-time per database version)
      yes, hint is small / fresh        → "ypir"        (USENIX'24, best comm)
                                          or "simple_pir" / "frodo_pir"
      no                                → "spiral"      (2-3s on 1e8, no hint)
```

Or skip the flowchart entirely and let the selector decide:

```bash
# Using the skill wrapper (recommended):
python3 skills/primihub-pir/cli.py auto-select \
    --db-size=1e8 --query-type=index --latency-budget=ms \
    --allow-two-server --assume-non-colluding --client-can-cache-hint \
    --dry-run

# Or directly invoking pir_inspect on a build host:
pir_inspect auto db-size=1e8 query-type=index latency-budget=ms \
    allow-two-server=true assume-non-colluding=true \
    client-can-cache-hint=true dry-run=true
```

---

## Cheat sheet

| algorithm    | servers   | hint       | comm (typical) | latency @ 1e8 | status      | paper            |
|--------------|-----------|------------|----------------|---------------|-------------|------------------|
| `id_pir`     | 1         | none       | small          | seconds       | ✅ real      | SealPIR'18       |
| `apsi`       | 1         | none       | medium         | seconds       | ✅ real (opt-in) | APSI USENIX'21 |
| `spiral`     | 1         | none       | ~26 KB         | 2-3 s         | 🚧 skeleton | USENIX'22        |
| `simple_pir` | 1         | client (per-DB) | ~121 KB    | sub-second    | 🚧 skeleton | USENIX'23        |
| `double_pir` | 2 non-collude | client+server (per-DB) | ~4 KB | ~10 ms     | 🚧 skeleton | USENIX'23        |
| `frodo_pir`  | 1         | client (per-DB) | ~64 KB     | ms class      | 🚧 skeleton | PETS'23          |
| `ypir`       | 1         | client (per-DB) | minimal    | sub-second    | 🚧 skeleton | USENIX'24        |

> 🚧 **skeleton** = registrar / capabilities / proto compat are in place,
> selector ranks them correctly, OnExecute returns FAIL. Real cryptographic
> kernel lands per algorithm in OpenSpec change tasks 4.4 / 5.5 / 7.x.

---

## Sending a task with an explicit algorithm

PIR task config is JSON, parsed into a `Task.params.param_map` on the wire.
The standard params keys (documented in `proto/common.proto`):

| key                    | type   | values                                         |
|------------------------|--------|------------------------------------------------|
| `algorithm`            | STRING | Registered name (`id_pir`, `spiral`, …)        |
| `latency_budget`       | STRING | `any` \| `seconds` \| `sub-second` \| `ms`     |
| `preferred_backend`    | STRING | `auto` \| `cpu` \| `avx2` \| `cuda`            |
| `assume_non_colluding` | INT32  | 0 \| 1 (must be 1 for two-server schemes)      |
| `hint_path`            | STRING | Pre-generated hint file path (per-DB)          |

Minimal example (`example/spiral_index_task.json`):

```json
{
  "task_type": "PIR_TASK",
  "params": {
    "algorithm":           { "type": "STRING", "value": "spiral" },
    "latency_budget":      { "type": "STRING", "value": "seconds" },
    "preferred_backend":   { "type": "STRING", "value": "auto" },
    "assume_non_colluding":{ "type": "INT32",  "value": 0 },
    "hint_path":           { "type": "STRING", "value": "" },
    "outputFullFilename":  { "type": "STRING", "value": "data/result/pir.csv" }
  },
  "party_datasets": {
    "SERVER": { "SERVER": "your_dataset_id" }
  }
}
```

On the wire, `algorithm` takes precedence over the legacy `pirType` int
(0 = ID_PIR / 1 = KEY_PIR). When both are absent the request is rejected
at `PirTask::InitOperator` with a clear log message. When `algorithm` is
unset and `pirType` is set, the `LegacyNameFor` shim translates the enum
to a registry name (`ID_PIR → "id_pir"`, `KEY_PIR → "apsi"`). This means
**every pre-multi-algo client keeps working unchanged**.

---

## When the selector says "no algorithm satisfies constraints"

The selector returns an empty winner list when no registered algorithm
matches the constraints. The CLI dry-run output shows the failure reason
per algorithm:

```text
algorithm    | passes | score   | comm_KB | fail_reasons
---------------------------------------------------------
double_pir   | no     | 0       | 4       | allow_two_server is false
spiral       | no     | 0       | 26      | perf_class exceeds latency_budget
ypir         | no     | 0       | 0       | hint required but client cannot cache
...
```

Common reasons + fixes:

- **"db_size far exceeds recommended_max_db_size"** — the algorithm's
  parameters were tuned for smaller databases. Either bump `db_size`
  expectations down or pick a bigger-DB algorithm (DoublePIR / YPIR).
- **"hint required but client cannot cache"** — pass
  `client_can_cache_hint=true` after arranging hint distribution (see
  [hint-lifecycle.md](hint-lifecycle.md)).
- **"perf_class exceeds latency_budget"** — relax the budget (or accept
  the higher-latency algorithm; e.g. SpiralPIR can't hit ms).
- **"allow_two_server is false"** — DoublePIR needs two cooperating
  servers; opt in only after confirming non-collusion is a valid
  assumption in your deployment ([threat-model.md](threat-model.md)).

---

## Backend selection (CPU / AVX2 / CUDA)

Each algorithm declares which backends its capability profile supports.
`preferred_backend=auto` asks the runtime to probe (`__builtin_cpu_supports`,
`cudaGetDeviceCount`) and pick the best available. Currently:

- `id_pir` — CPU only (SealPIR fork doesn't have AVX2 path here)
- skeletons — CPU only profile until phase 7 lands CUDA kernels

Force a backend with `preferred_backend=cpu|avx2|cuda` when debugging
performance regressions or running mixed-build comparisons.

---

## Adding a new algorithm

1. Create `src/primihub/kernel/pir/operator/<algo>/{<algo>.h,<algo>.cc,BUILD}`
2. Define a `PirCapabilities` profile (query types, server count, hint,
   perf class, threat model, recommended max db, backends, typical
   comm/hint bytes).
3. Add a `PirRegistrar<YourOp> reg_("<name>", caps);` at namespace scope
   in the cc file (anonymous namespace).
4. Set `alwayslink=True` in the BUILD target so the registrar runs even
   when no other code references the algorithm directly.
5. Add `cc_test` entries in `src/primihub/kernel/pir/tests/`. Use the
   skeleton tests in `spiral_pir_test.cc` / `double_pir_test.cc` as the
   pattern.
6. Pull the new target into `src/primihub/cli/pir_inspect/BUILD` so the
   CLI shows it in `list`.
7. Update [benchmark.md](benchmark.md) and this guide once the registrar
   moves from skeleton → real.

---

## Workflow recipes

### Quick "is X better than Y for my workload?"

```bash
# Spit out caps JSON for two algos side-by-side
diff -u <(pir_inspect caps spiral) <(pir_inspect caps double_pir)
```

### Lock in a chosen algorithm in CI

```bash
ALGO=$(pir_inspect auto db-size=1e8 query-type=index latency-budget=sub-second \
    client-can-cache-hint=true 2>/dev/null)
echo "ci will use PIR algorithm: $ALGO"
# pipe ALGO into your task config generator
```

### Catch unintended selector drift

Run [bench/pir_selector_sweep.sh](../../bench/pir_selector_sweep.sh) and
diff against the last committed JSON in `bench/results/`. Cells whose
`winner` changed identify constraints where ranking shifted; investigate
whether that's an intentional improvement (then commit the new JSON) or
a regression (then bisect and revert).

---

## See also

- [threat-model.md](threat-model.md) — per-algorithm security assumptions
- [hint-lifecycle.md](hint-lifecycle.md) — managing per-DB hint files
- [activation-pattern.md](activation-pattern.md) — runtime facade contract for new operator ports
- [benchmark.md](benchmark.md) — captured benchmark + selector results
- `bench/pir_runtime_activations.sh` — unified CI signal for all activated operators
- `openspec/changes/primihub-pir-multi-algo/` — original OpenSpec change
