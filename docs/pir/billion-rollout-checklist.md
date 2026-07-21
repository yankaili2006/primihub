# PIR Billion-scale Rollout — execution checklist

The PIR **engineering** is done and merged (framework + 5 algorithms + TipToe +
CUDA backends + the SpiralPIR AVX2 fix; 71/71 regression green on the local AVX512
box, primihub `45e8d876`). What remains is **deployment/ops + the `pir_type`
migration**, all gated on infrastructure that doesn't exist yet (a deployed
billion-scale primihub node + production traffic over calendar time).

This is the runbook to execute when that infra is available. It mirrors the
openspec change `primihub-pir-billion-rollout` (currently archived/DEFERRED at
`openspec/changes/archive/2026-06-27-primihub-pir-billion-rollout/`, pcloud repo)
and the canonical `pir-migration` spec (`openspec/specs/pir-migration/spec.md`).

## 0. Re-open the openspec change first

- [ ] Move it back to active:
      `git mv openspec/changes/archive/2026-06-27-primihub-pir-billion-rollout \
             openspec/changes/primihub-pir-billion-rollout` (in the **pcloud** repo)
- [ ] Flip the `tasks.md` banner ARCHIVED → RE-OPENED; check boxes below off there
      as each lands. Re-archive (with a date prefix) when all 5 are done.

## Phase 1 — Billion-scale benchmark (change tasks 1.1, 1.2)

Prereqs: a reachable primihub node with a 1e8 (then 1e9) DB loaded server-side.
A GPU host makes the CUDA backend measurable but is optional for correctness.

- [ ] **1.1 Stand up the node + confirm client flags.** Deploy a primihub node,
      load a 1e8 DB. Confirm the exact `primihub-cli` PIR task-run subcommand/flags
      and wire them into `bench/pir_billion_scale_bench.sh` `run_one_cell()` at the
      `TODO(deploy)` marker. **Ideally** make the client emit per-phase
      offline/online timings (so the harness reports real setup vs per-query
      instead of splitting wall-clock).
- [ ] **1.2 Run the bench for real + commit results.**
      `bench/pir_billion_scale_bench.sh --node <host:port> \
         --algorithm "spiral,double_pir" --n 100000000 --trials 3 \
         --out bench/results/pir_billion_scale_1e8.json`
      then repeat with `--n 1000000000`. Commit the resulting
      `bench/results/pir_billion_scale_*.json` as the known-good snapshot, and add
      the numbers to `docs/pir/benchmark.md` (a "deployed-node" section next to the
      standalone CUDA matvec table).
      - Sanity vs the standalone Answer-kernel floor already measured on the local
        RTX 5070 Ti (`docs/pir/benchmark.md`): 1e9 CUDA warm **15.14 ms / 24.7×**.

## Phase 2 — pir_type → algorithm migration (change tasks 2.0, 2.1, 2.2)

Implements the canonical `pir-migration` spec (3 requirements). The spec is already
the agreed contract; these tasks make it real, so flip its `## Purpose`
"pending implementation" note once done.

- [ ] **2.0 Dual-run shim + telemetry** (reqs *Dual-run acceptance* + *Legacy
      telemetry*). On the PIR task request, accept BOTH legacy `pir_type` and new
      `algorithm`; route `pir_type` through the registry's backward-compat shim
      (the shim already exists per `pir-registry` spec — wire request acceptance to
      it). Precedence: `algorithm` wins; **reject** requests where the two resolve
      to different algorithms with an explicit error. Emit
      `pir_legacy_pir_type_total` (counter, labelled by resolved algorithm) + a
      structured log line on every legacy-field request.
- [ ] **2.1 30-day dual-run monitoring** (req *Legacy telemetry*, cont.). Collect
      `pir_legacy_pir_type_total` over a ~30-day window; quantify the share of
      remaining legacy (`pir_type`-only) clients vs new (`algorithm`) clients.
- [ ] **2.2 Deprecation timetable** (req *Deprecation timetable*). From the 2.1
      telemetry, publish a doc with the dates `pir_type` moves to deprecated
      (accepted + warned) and to removed. During the deprecated phase, log a
      deprecation warning per legacy request naming the removal date + the
      `algorithm` to switch to. Keep the shim functional until the removal date.

## Acceptance (close the change when all true)

- [ ] Committed `bench/results/pir_billion_scale_*.json` with real (non-dry-run)
      cells for `spiral` + `double_pir` at N ≥ 1e8.
- [ ] `pir-migration` requirements satisfied: dual-run shim live, legacy telemetry
      collected, dated migration timetable published.
- [ ] All 5 boxes in the re-opened `tasks.md` checked; change re-archived.

## Out-of-band (not infra-gated — do anytime)

- [ ] **Rotate the exposed GitHub PAT.** A classic PAT (account `yankaili2006`,
      `repo` scope) was exposed. As of 2026-06-27 the stored token already returns
      **HTTP 401** (expired/revoked) — no longer exploitable, but Vault
      `secret/pcloud/github` holds a dead credential to replace. Rotation now runs
      through the **vault skill** (paste-safe input + pre-write GitHub validation +
      in-Vault audit log), not ad-hoc scripts:
      1. Browser: https://github.com/settings/tokens -> Generate new token
         (classic) -> tick ONLY `repo` -> copy. (GitHub has no PAT-creation API.)
      2. `cd <pcloud>/skills/vault && python3 skill.py rotate pcloud/github \
            --field token --validate github --expect-user yankaili2006 \
            --note "classic PAT repo scope (rotated <date>)"`  (paste when prompted)
      3. Browser: revoke the old token.
      4. `python3 skill.py rotation-log pcloud/github` to confirm the audit entry;
         re-run any consumer to confirm the new token works.

## Optional follow-ups (perf, non-blocking)

- [ ] CUDA: reuse `pir-acc/SIGMA`'s tuned NTT + device-resident data to lift the
      warm GPU numbers (current kernels are correctness-first; 24.7× at 1e9 is a
      floor). SpiralPIR GSW/Galois + DoublePIR matmul kernels live in
      `src/primihub/kernel/pir/operator/{spiral_pir,double_pir}/cuda/`.

## Pointers

- openspec change: `openspec/changes/archive/2026-06-27-primihub-pir-billion-rollout/` (pcloud)
- spec: `openspec/specs/pir-migration/spec.md` (pcloud)
- bench harness: `bench/pir_billion_scale_bench.sh` (dry-run-verified)
- standalone GPU numbers: `docs/pir/benchmark.md` (cuda_vs_avx2, 1e8 + 1e9)
- build/test env + gitcode spiral pin: see the pcloud PIR memory notes.
