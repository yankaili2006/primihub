# PIR Hint Lifecycle

Several algorithms in the multi-algo framework — SimplePIR, FrodoPIR,
YPIR, and DoublePIR — require the client to hold a **hint** before any
query can be issued. Hints are not secret (they contain only public
projections of the database) but they ARE bound to a specific database
version. This document explains how primihub manages hint generation,
distribution, and invalidation.

---

## What's a hint, mechanically?

For LWE-based PIR (SimplePIR / FrodoPIR / YPIR), the hint is a
**pre-computation of A · D**, where:

- `A` is a public matrix sampled deterministically from a per-database
  seed,
- `D` is the database itself, packed as a matrix of `Z_q` entries.

The client uses the hint to short-circuit the most expensive part of the
online phase. Without the hint, the client would have to either ship
the full database (defeating PIR) or do online linear work proportional
to N (defeating the latency win).

For DoublePIR, the hint takes the form of pre-computed partial responses
that both servers ship to the client during the offline phase.

---

## Implementation status (DoublePIR, task 5.6 chunks 1-5)

The single-process side of hint lifecycle is implemented and merged on
the `pir-multi-algo` branch:

| Chunk | Commit     | What landed                                                                 |
|-------|------------|-----------------------------------------------------------------------------|
| 1     | `7303a83e` | `HintGen::Compute` lifts Init+Setup out of `OnExecute` into a reusable static function; bundles A1/A2/H1_squished/A2_copy_transposed/H2_msg/info_after_setup into `DoublePirHint`. |
| 2     | `7b017575` | Process-local LRU `HintCache` (cap=16) keyed by `(l, m, p, logq, FNV-1a(DB))`. Hit skips O(L·M·n) Setup; caller re-runs cheap O(L·M) `ScalarAdd(p/2) + Squish(10,3)` to restore squished-DB shape. |
| 3     | `3595f95f` | Wire format `SerializeHint` / `DeserializeHint` (PHHB magic + u16 version + 10·u64 DBinfo + 5 matrices). 168 B fixed overhead. Guards against bad magic / unsupported version / dim overflow / truncation / trailing bytes. |
| 4     | `5e9630ee` | `HintCache::SaveToFile` / `LoadFromFile` wrap PHHB blobs in a PHHC outer envelope. Atomic write via `<path>.tmp` + rename. Load stages framing checks then `Clear()` + `Put()` so malformed files never clobber existing state. |
| 5     | `59333cad` | `DoublePirOperator::OnExecute` auto-loads from `options_.hint_path` (populated by `pir_task.cc` from the `hint_path` param) via idempotent `MaybeLoadOnce`, auto-saves after a successful cache miss. Errors are LOG(WARNING) only — queries never fail because persistence misbehaved. |

What this means for callers: set `hint_path` in the task's
`param_map` and the operator transparently persists hints across
process restarts. No out-of-band CLI invocation needed for the
common case; the "out-of-band cron-style refresh" pattern below
still applies for **cross-machine** hint distribution where a single
server precomputes hints and ships them to many clients.

### Measured speedup (DoublePIR, .50, queries=4)

| N    | cold setup_ms | warm setup_ms | wall speedup |
|------|---------------|---------------|--------------|
| 64   | 43            | 0             | 1.6×         |
| 256  | 60            | 0             | 1.7×         |
| 1024 | 97            | 0             | 1.8×         |
| 4096 | 224           | 0             | 3.4×         |

Reproduce via `bench/double_pir_persistence_bench.sh`. Speedup grows
with N because cold Setup is O(L·M·n); warm Setup is fixed at 0.
At 1e8 the paper's ~110 s Setup vs ~70 ms per-query yields >1000×
warm-start wall speedup.

### Ops verification

`pir_inspect cache <path> [--emit-csv]` (commit `a41d3d1e`) reads a
PHHC file and prints per-entry summaries — useful for verifying
what's been persisted before restarting a production node.

### Remaining work

- **5.6 LinkContext peer-split** — two non-colluding servers each
  compute their slice of the hint locally and ship via
  `MultiPeerPirOperator::SendToAllPeers` using the PHHB wire format.
  Blocked on the production CLIENT/SERVER LinkContext design.
- **Other algorithms** — only DoublePIR's hint lifecycle is wired
  through chunks 1-5 so far. SimplePIR/FrodoPIR/YPIR will pick up
  the same `Options.hint_path` + `MaybeLoadOnce` pattern when their
  Setup paths are de-inlined to match.

---

## Three lifecycle phases

```text
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  Generation  │───▶│ Distribution │───▶│  Online use  │
   │   (server)   │    │              │    │   (client)   │
   └──────┬───────┘    └──────────────┘    └──────┬───────┘
          │                                        │
          │            DB rev change               │
          └────────────────────────────────────────┘
                       triggers regeneration
```

### Generation

Server-side, by the same `BasePirOperator` that will later answer queries.
Triggered by passing `generate_db=true` in the task config (currently the
SealPIR path uses this) or by an out-of-band cron-style refresh.

The operator writes:

- `<db_path>/hint.bin` — the binary hint blob
- `<db_path>/hint.meta.json` — sidecar with hint metadata (see schema below)

### Distribution

Hints are public — no encryption required — but integrity matters: a
swapped hint can sometimes degrade query privacy depending on the
specific scheme. primihub's distribution model:

- **OSS** — default. Hints land in
  `oss://primihub/pir/hints/<algo>/<db_id>/<version>/hint.bin`.
  Public-read ACL, SHA-256 checksum recorded in the OSS object
  metadata and re-verified by the client on download.
- **Co-located** — for single-datacenter deployments, a shared NFS
  mount works (the operator config sets `hint_path` to the NFS path
  and skips the client download step entirely).
- **Push to client** — for clients that connect intermittently, the
  server pushes the hint over the existing gRPC channel during the
  PIR task setup phase, before the first query.

The hint file MUST be served over an integrity-protected channel (TLS +
checksum verification). A malicious upstream that swaps a hint can
degrade query privacy depending on the specific scheme.

### Online use

Client reads `hint_path` from its local cache (or the path set in the
task config), checks the meta sidecar's `db_version` against what it
expects, and feeds the hint into the operator's `OnExecute` along with
the actual query.

If the client's cached hint disagrees with the server's current DB
version, the request fails with a "hint stale" error and the client
must download a fresh hint before retrying.

---

## hint.meta.json schema

```json
{
  "schema_version": 1,
  "algorithm": "simple_pir",
  "db_id": "primihub-platform.user_attrs",
  "db_version": "2026-06-06T08:00:00Z",
  "db_hash": "sha256:...",
  "params": {
    "n": 2048,
    "q": 9223372036854775783,
    "...": "algo-specific parameter set"
  },
  "hint_sha256": "...",
  "hint_size_bytes": 67108864,
  "generated_at": "2026-06-06T08:01:23Z",
  "generated_by": "primihub-node@fusion0.example.com",
  "ttl_hint": "7d",
  "next_refresh_after": "2026-06-13T08:00:00Z"
}
```

- `db_version` — opaque token (timestamp / monotonic counter / git sha)
  the server uses to declare "this hint matches this DB content".
- `db_hash` — SHA-256 of the canonical DB representation. Defence in
  depth: even if `db_version` is wrong, the hash check catches it.
- `ttl_hint` / `next_refresh_after` — advisory. Clients SHOULD refresh
  past this point even if the server hasn't said the hint is stale,
  because cryptographic parameters may have been retired upstream.

---

## When to regenerate

Hint regeneration is **mandatory** when:

1. **Any row's content changes.** Insertions, updates, deletions, even
   schema migrations that reorder columns — all invalidate the hint.
2. **The algorithm's parameter set changes.** Bumping LWE dimension n
   or modulus q invalidates the hint even when the DB is unchanged.
3. **The algorithm's `recommended_max_db_size` is exceeded.** Past this
   point the parameters may no longer give the security level the
   profile claims; regenerate with a new parameter set.

It is **advisory** when:

- The `ttl_hint` window has elapsed but the DB hasn't changed.
- A new primihub-node version has shipped (hints sometimes pick up
  cryptographic-library bugfixes that don't break compat but are nice
  to refresh).

---

## Bandwidth and storage

Hints can get large:

| algo         | hint size for 1e6 rows | hint size for 1e8 rows |
|--------------|------------------------|------------------------|
| `simple_pir` | ~12 MB                 | ~1.2 GB                |
| `frodo_pir`  | ~25 MB                 | ~2.5 GB                |
| `ypir`       | ~3 MB                  | ~300 MB                |
| `double_pir` | ~256 KB                | ~26 MB                 |

For 1e8 deployments, the hint distribution channel matters: shipping a
1.2 GB hint to each client over the public internet is a real cost.
Mitigations:

- **CDN-fronted OSS** so geographically-distributed clients hit nearby
  cache nodes.
- **Differential hints** — when the underlying DB changes by a small
  delta, ship only the changed rows' contribution and let the client
  update the cached hint locally. This is an open research direction;
  primihub doesn't implement it today.
- **In-fusion hint storage** — fusion nodes hold the hint and clients
  only download it once on first connection.

---

## Failure modes

| symptom                          | cause                              | fix                                     |
|----------------------------------|------------------------------------|-----------------------------------------|
| "hint required but client cannot cache" in selector dry-run | `client_can_cache_hint=false` | Set `--client-can-cache-hint=true` only after arranging distribution |
| "hint stale" at query time       | DB version drifted                 | Re-download hint, retry query           |
| Query returns garbage            | Hint integrity broken silently     | Always verify `hint_sha256` on download |
| Hint download times out          | Large blob, narrow channel         | Switch to CDN-fronted OSS or co-located storage |
| `HintCache::MaybeLoadOnce` WARNING in node log | Persisted PHHC file missing / corrupt / on read-only fs | Advisory only — operator continues with an empty cache and saves a fresh PHHC after the first miss. Inspect with `pir_inspect cache <path>` to confirm format. |
| `HintCache::SaveToFile` WARNING after query | Disk full / read-only fs / permission error on `<hint_path>.tmp` | Advisory only — query still returns the answer. Check disk + perms; the next OnExecute will retry the save. |

---

## Hooking hint generation into your DB pipeline

primihub-platform's current data pipeline doesn't run hint generation
automatically. When operationalizing a hint-requiring algorithm:

1. Add a post-ingest hook that calls
   `primihub-cli --task_config_file=hint_generate_<algo>.json`
2. Publish the generated hint + meta to your distribution channel.
3. Notify connected clients via the existing gRPC notification stream
   so they refresh their cached hint.

For shipping schedules, treat hints like database schema migrations —
both server and client need to be coordinated, and rollback means
keeping the previous hint version reachable for a transition window.

---

## See also

- [multi-algo-guide.md](multi-algo-guide.md) — when to pick a
  hint-requiring algorithm in the first place
- [threat-model.md](threat-model.md) — hint integrity threat surface
- `src/primihub/kernel/pir/operator/base_pir.h` — `Options.hint_path`
- `src/primihub/kernel/pir/operator/double_pir/hint_cache.{h,cc}` —
  process-local LRU + `MaybeLoadOnce` / `SaveToFile` / `LoadFromFile`
- `src/primihub/kernel/pir/operator/double_pir/hint_serialize.{h,cc}` —
  PHHB wire format
- `src/primihub/cli/pir_inspect/pir_inspect_main.cc` — `cache`
  subcommand for ops verification
- `bench/double_pir_persistence_bench.sh` — cold-vs-warm latency bench
- OpenSpec change `primihub-pir-multi-algo`, tasks 5.6 (`hint_gen`)
  and 7.x (per-algo hint formats)
