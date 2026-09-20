# PIR Multi-Algorithm Guide

> Status: **6 algorithms registered (3 real + 3 skeletons; 3 of the skeletons have C/C++ kernel link activated)** as of 2026-06-08.
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
| `simple_pir` | 1         | client (per-DB) | ~121 KB    | sub-second    | ✅ real      | USENIX'23        |
| `double_pir` | 2 non-collude | client+server (per-DB) | ~4 KB | ~10 ms     | ✅ real      | USENIX'23        |
| `frodo_pir`  | 1         | client (per-DB) | ~64 KB     | ms class      | ✅ real      | PETS'23          |
| `ypir`       | 1         | client (per-DB) | minimal    | sub-second    | 🚧 skeleton | USENIX'24        |

> 🚧 **skeleton** = registrar / capabilities / proto compat are in place,
> selector ranks them correctly, OnExecute returns FAIL. Real cryptographic
> kernel lands per algorithm in OpenSpec change tasks 4.4 / 7.3.
>
> ✅ **real** algorithms: id_pir (SealPIR, since project inception), apsi
> (opt-in via --define microsoft-apsi=true), simple_pir (task 7.2 landed
> 2026-06-08 / commit 2d77509a), double_pir (task 5.5 landed 2026-06-08 /
> commit 31e8fd43 — first single-process self-contained pipeline; the
> production CLIENT/SERVER LinkContext split is task 5.6).

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

## Hint persistence (`hint_path`)

Both **DoublePIR** and **SimplePIR** now persist hints across process
restarts when `hint_path` is set. The wire-level mechanics are the
same for both algorithms; only the wire magic differs (PHHC for
DoublePIR, PSHC for SimplePIR).

What happens during `OnExecute`:

1. **First call per (process, path)** — `HintCache::MaybeLoadOnce(path)`
   attempts to load any persisted cache file. Missing/corrupt files
   surface as a `WARNING` in the node log; the operator continues
   with whatever cache state it already had.
2. **Cache miss + successful compute** — `HintCache::SaveToFile(path)`
   persists the new hint, atomically via `<path>.tmp` + rename. Save
   failures (disk full, permission errors) are also advisory.
3. **Cache hit** — skips the O(L·M·n) Setup multiply entirely;
   caller re-applies the cheap `ScalarAdd(p/2) + Squish(10, 3)`
   locally to restore the squished-DB shape Answer expects.

Errors from persistence never fail the query. On-call sees
`WARNING`s; the actual PIR work always completes if the operator
can compute the hint itself.

### When to set `hint_path`

- Production deployments that restart frequently (rolling upgrades,
  k8s pod reschedules, OOM kills) — cold-start latency drops to
  warm-start levels after the first miss.
- Single-shot queries can leave `hint_path` empty; the in-process
  LRU still amortizes across multiple OnExecute calls within the
  same process.
- Hint files are bound to a specific `(l, m, p, logq, FNV-1a(DB))`
  fingerprint. Changing the DB invalidates the cache silently —
  the next call hashes to a different fingerprint and misses, then
  overwrites the file with the fresh hint.

### Ops verification

```bash
pir_inspect cache /var/cache/primihub/double_pir_hints.bin
# PHHC HintCache (DoublePIR) @ /var/cache/primihub/double_pir_hints.bin
#   version : 1
#   entries : 4
# entry[0] fp=0x… blob=… B | A1=… A2=… H1sq=… A2copyT=… H2msg=… | cells=… info{…}

# Same CLI auto-detects PSHC (SimplePIR) files.
pir_inspect cache /var/cache/primihub/simple_pir_hints.bin
```

### Measured speedup

See `docs/pir/benchmark.md` for the full cold-vs-warm tables. Headline
numbers on .50 (queries=4):

| algorithm  | N=4096 wall speedup | N=4M wall speedup |
|------------|---------------------|-------------------|
| DoublePIR  | 3.4×                | (memory-limited)  |
| SimplePIR  | 2.7×                | 3.7×              |

Reproduce via `bench/double_pir_persistence_bench.sh` and
`bench/simple_pir_persistence_bench.sh`.

### What's *not* yet wired

- **FrodoPIR / YPIR / Tiptoe** are still skeletons; their operators
  don't yet consume `hint_path` because they don't yet have real
  Setup paths. When they land they'll pick up the same pattern.

---

## Two-peer hint distribution (`hint_role`) — DoublePIR only

DoublePIR's threat model assumes two **non-colluding** servers. Both
hold the database; both can independently derive the same hint locally
(generation is deterministic given the DB). Running `HintGen` on both
servers wastes O(L·M·n) Setup work that only one server actually
needs to perform.

`Options.hint_role` (populated from `param_map["hint_role"]`) lets one
peer act as the **primary** (computes the hint, broadcasts it) and the
other as the **secondary** (receives the hint over `LinkContext` in
lieu of `HintGen`).

| `hint_role` value | Behaviour                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------|
| (empty, default)  | Single-process — operator runs the full protocol locally, no wire I/O regardless of `peer_nodes`. |
| `"primary"`       | Local `GetOrComputeHint` then `BroadcastHint` to every node in `peer_nodes`. Broadcast failure is `LOG(WARNING)` only — the primary's own query still completes. |
| `"secondary"`     | `ReceiveHint` from `peer_nodes[0]` (the primary) replaces `HintGen`. Result is `Put()` into the local `HintCache` so a same-process re-run hits the LRU. Local DB still goes through the cheap `ScalarAdd(p/2) + Squish(10, 3)` pair so the squished-DB shape matches what Answer expects. |

Wire payload is the same PHHB blob used by `hint_path` persistence —
serialization is shared between disk and network. Key on the wire is
`"double_pir.hint.v1"`; override via the `key` argument to
`BroadcastHint` / `ReceiveHint` if you need to namespace multiple
concurrent hints in flight.

### Task config example (`param_map`)

Primary peer's task config:

```json
{
  "algorithm":   { "type": "STRING", "value": "double_pir" },
  "hint_role":   { "type": "STRING", "value": "primary" },
  "hint_path":   { "type": "STRING", "value": "/var/cache/primihub/dp_hint.bin" }
}
```

Secondary peer's task config:

```json
{
  "algorithm":   { "type": "STRING", "value": "double_pir" },
  "hint_role":   { "type": "STRING", "value": "secondary" }
}
```

Both peers also need `peer_nodes` populated by the scheduler — the
secondary expects `peer_nodes[0]` to be the primary, the primary
expects `peer_nodes` to enumerate all secondaries to broadcast to.

`hint_path` and `hint_role` compose: a primary may also persist its
hint to disk for cold-restart speedup, and a secondary may also keep
the received hint in its on-disk cache. Most production deployments
set both on the primary and only `hint_role="secondary"` on the
secondary.

### Failure modes

| Symptom                                                          | Cause                                                                  | Fix                                                                                |
|------------------------------------------------------------------|------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Operator returns FAIL with `hint_role="primary" with non-empty peer_nodes requires Options.link_ctx_ref` | Caller set `peer_nodes` but never plumbed `link_ctx_ref`. | Make sure the scheduler injects `Options.link_ctx_ref` before constructing the operator. |
| `LOG(WARNING): BroadcastHint: Send to peer index N failed`       | Network partition or secondary not listening on the wire key.          | Operator continues without breaking the primary's own query. Investigate the peer.        |
| Operator returns FAIL with `ReceiveHint: Recv from peer id=... failed` | Secondary started before primary's broadcast arrived, OR primary failed to broadcast. | Retry the task. If the primary repeatedly fails the broadcast (check `WARNING` logs there) treat it as a primary-side outage. |
| Operator returns FAIL with `hint_role="secondary" requires peer_nodes[0]` | Secondary task config forgot to declare the primary in `peer_nodes`. | Add the primary's `Node` to `peer_nodes` in the secondary's config.                  |

### When to use which mode

- **Single-shot demos / local development** — leave `hint_role`
  empty. Single-process is fine and the test surface is smallest.
- **Pre-production / multi-node smoke** — both servers run with
  `hint_role` set. Verifies the wire path end-to-end.
- **Production deployment** — primary holds the authoritative hint
  (often also persisted via `hint_path`); secondaries pick up the
  hint over the wire on each task. Saves O(L·M·n) Setup per secondary
  per cold cache.

### Cross-machine hint distribution

Beyond the per-task `hint_role` split above, the "one server
precomputes, many clients download" pattern is the out-of-band
cron-style refresh story documented in
`docs/pir/hint-lifecycle.md` — orthogonal to the per-process
persistence and per-task wire transport covered here.

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
