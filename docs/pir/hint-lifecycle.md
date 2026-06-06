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
- OpenSpec change `primihub-pir-multi-algo`, tasks 5.6 (`hint_gen`)
  and 7.x (per-algo hint formats)
