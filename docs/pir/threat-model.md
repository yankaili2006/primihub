# PIR Threat Model

Each PIR scheme in primihub assumes a specific adversary model. Choosing
the right algorithm requires confirming your deployment matches its
assumptions — picking DoublePIR for a scenario where both servers can
collude **silently destroys query privacy** (the very property PIR exists
to protect), and no protocol layer above will detect this.

This document captures the assumptions, known attacks, and operational
guidance for every algorithm currently registered or planned in the
multi-algo framework.

---

## Common framing

All algorithms in primihub today:

- Protect **query privacy** — the server learns nothing about *which*
  rows the client retrieves. This is the asymmetric guarantee PIR exists
  to provide.
- **Do not protect database confidentiality from the client** — the
  client learns the row(s) it queried, in plaintext, by construction.
  If the database is sensitive, scope it accordingly (e.g. authorization
  before query, or compose with secure aggregation downstream).
- Assume the underlying lattice / FHE primitives are post-quantum hard
  per their published parameters. Down-bumping security parameters to
  hit a latency target is outside the framework's safe defaults.

The `ThreatModel` enum (`src/primihub/kernel/pir/common.h`) captures the
adversary assumption for the *PIR scheme itself*:

| ThreatModel               | Server behaviour                                  |
|---------------------------|---------------------------------------------------|
| `SemiHonest`              | Server follows the protocol, tries to infer query |
| `SemiHonestNonColluding`  | Multiple servers, each semi-honest, do not share state |
| `Malicious`               | Server may deviate arbitrarily (not yet supported) |

The selector refuses to recommend an algorithm whose `threat_model` is
weaker than the user's `min_threat_model` constraint.

---

## Per-algorithm assumptions

### `id_pir` — SealPIR (USENIX'18)

- **Assumption:** Single server, semi-honest. Server is allowed to
  observe the encrypted query ciphertext and the resulting ciphertext
  arithmetic; security reduces to the underlying RLWE hardness with
  SEAL 4.0's default parameters.
- **Known attacks:** None practical at default parameters. Side-channels
  on the server's response timing have been published but are addressed
  by the FHE library, not at the PIR layer.
- **Operational notes:** Database size envelope is small (~1e6 rows
  before query latency becomes painful even on modern CPUs). Don't try
  to push 1e8 unless you have substantial RAM headroom and accept
  10-30s per-query latency.

### `apsi` — Microsoft APSI Keyword PIR (USENIX'21)

- **Assumption:** Single server, semi-honest. Built on top of SealPIR
  primitives, so same RLWE assumption. The keyword-mapping layer adds
  a Cuckoo hash and OPRF; OPRF security relies on the server holding
  a secret key that the client never sees.
- **Known attacks:** Set-size leakage. Server learns the **count** of
  client queries (per batch). If batches are small, this can be a
  practical fingerprint; pad batches to a fixed multiple to mitigate.
- **Operational notes:** Builds in only when `--define microsoft-apsi=true`
  was passed to bazel. Keyword set must be pre-processed; refresh on
  schema change.

### `spiral` — SpiralPIR (USENIX'22) 🚧 skeleton

- **Assumption (planned):** Single server, semi-honest. Uses GSW-style
  ciphertexts with Galois automorphisms; security reduces to RLWE.
- **Known attacks:** None published at the suggested parameter sets.
- **Operational notes:** No client-side hint required — same trust
  envelope as SealPIR but ~10× faster on the same hardware.

### `simple_pir` — SimplePIR (USENIX'23) 🚧 skeleton

- **Assumption (planned):** Single server, semi-honest. Plain LWE.
- **Known attacks:** None practical.
- **Operational notes:** Hint is **per-database-version**, so any change
  to row content (not just additions) invalidates it. The hint contains
  no per-client state, so a single hint serves all clients on the same
  DB version.

### `frodo_pir` — FrodoPIR (PETS'23) 🚧 skeleton

- **Assumption (planned):** Single server, semi-honest. Plain LWE with
  Frodo-style parameters (no ring structure → conservative).
- **Known attacks:** None published.
- **Operational notes:** Industrial maintenance by Brave; slightly
  larger hint than SimplePIR but easier parameter justification.

### `double_pir` — DoublePIR (USENIX'23) 🚧 skeleton

- **Assumption (planned):** **TWO servers, semi-honest, non-colluding.**
  This is the strongest operational assumption in the framework.
  Security guarantees collapse to no privacy if the two servers share
  state, run on the same machine, or are operated by the same legal
  entity that can be compelled to combine logs.
- **Known attacks:** None against the protocol when assumption holds.
  When the assumption fails: combining the two servers' query traces
  reveals the queried index in plaintext.
- **Operational guidance:**
  - Selector and `PirTask::InitOperator` both require
    `assume_non_colluding=1` to be passed explicitly. There is no
    default; the field must be set by an operator who has independently
    verified the non-collusion property.
  - In primihub's fusion-node deployment model, this means the two
    `fusion0` / `fusion1` nodes **must be operated by different
    organizations** with no shared infrastructure (different cloud
    accounts, different physical hosts, different on-call paging).
  - Audit log retention should be configured so neither party can
    retroactively reconstruct the other's view.

### `ypir` — YPIR (USENIX'24) 🚧 skeleton

- **Assumption (planned):** Single server, semi-honest. LWE-based.
- **Known attacks:** None at suggested parameters.
- **Operational notes:** Best-in-class communication cost; hint is
  per-DB. Library is research-grade — keep `--define disable_ypir=1` as
  a tested fallback in case the upstream pivots.

### `tiptoe_pir` — TiptoeIR (SOSP'23) 🚧 unreleased

- **Assumption (planned):** Single server, semi-honest. Semantic keyword
  matching via embedding similarity.
- **Operational notes:** Hardest to reason about because the "query"
  semantics are approximate; threat surface includes the embedding
  space (a client whose embeddings are close to a target's reveals
  partial information by query patterns). Not in scope for the initial
  multi-algo rollout.

---

## What about Malicious adversaries?

`ThreatModel::Malicious` exists in the enum so future work doesn't have
to break the API, but no algorithm currently provides malicious security.
Upgrading to malicious would require either:

- Replacing the PIR scheme with a constructively-verifiable variant
  (e.g. TreeMVR-PIR), which trades 5–50× overhead, or
- Composing the current scheme with a non-interactive zero-knowledge
  proof of correct server execution, which is heavy enough that no
  open-source PIR system does this today.

If your deployment genuinely needs malicious security, do not enable PIR
on top of an untrusted server — use the TEE-PSI / SGX-attested execution
path instead (`primihub-tee` module).

---

## Checking your deployment

Before going to production with any algorithm, verify:

1. **Server count matches the algorithm's `min_servers` / `max_servers`.**
   Especially: don't run DoublePIR on a single physical host even if you
   split it into two processes — that *is* collusion.
2. **`assume_non_colluding` is set in your task config** if and only if
   the deployment really satisfies it. Spurious "true" is the most
   dangerous default-overrideable setting in the framework.
3. **Hint distribution is integrity-protected.** A malicious adversary
   that swaps a fake hint can sometimes degrade privacy; ship hints
   through a channel with TLS + signed checksums, not plain HTTP.
4. **Algorithm's `ThreatModel` matches your `min_threat_model`.** The
   selector enforces this when called via the CLI, but custom callers
   constructing `Options` directly need to check by hand.

---

## See also

- [multi-algo-guide.md](multi-algo-guide.md) — choosing an algorithm
- [hint-lifecycle.md](hint-lifecycle.md) — hint trust + distribution
- [benchmark.md](benchmark.md) — empirical performance numbers
