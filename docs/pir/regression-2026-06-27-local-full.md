# PIR full regression — 71/71 PASS on local AVX512 box (2026-06-27)

First **complete** PIR regression: all 71 `//src/primihub/kernel/pir/tests/...`
targets pass, including the two APSI/keyword targets that are unbuildable on `.50`
(their transitive externals — SEAL/Kuku/flatbuffers/GSL/cityhash — are GFW-blocked
there). Run on the local box (Xeon w/ **AVX512**, not GFW-blocked, reaches both
github and gitcode).

## Command

```bash
cd /mnt/sda/work/pir-multi-algo            # master @ 69b3c230 (gitcode spiral pin)
bazel test --config=linux_x86_64 --define=enable_spiral_real=1 \
  //src/primihub/kernel/pir/tests/... --keep_going --nocache_test_results
# -> Executed 71 out of 71 tests: 71 tests pass.
```

`@spiral_pir` is fetched (no `--override_repository`) from the self-maintained
gitcode fork `gitcode.com/yankaili2006/spiral @1d9c0f34` (AVX2 fix baked in); the
real-mode `spiral_runtime_test` prints **`Is correct?: 1`**.

## Coverage (71 targets)

- **Framework:** backend, capabilities, cli, database, registry, selector,
  multi_peer, matrix, gaussian, proto_compat
- **SimplePIR:** operator, protocol, runtime, hint_cache, serialize_persist
- **DoublePIR:** test, protocol, role, runtime, hint_link
- **FrodoPIR:** pir, api, database, flat_matrix, format, lwe_consts, matrices,
  params, prng
- **SpiralPIR:** spiral_pir, spiral_params, **spiral_runtime (real, AVX512,
  Is correct?: 1)**
- **TipToe:** tiptoe_pir, tiptoe_limb, tiptoe_secret
- **YPIR:** 36 targets (arith/bits/chacha/client/convolution/discrete_gaussian/
  e2e/e2e_large/gadget/kernel/lwe*/matmul/modulus_switch/negacyclic/operator/
  packing*/params/poly_ops/regev/runtime/scheme/server*/spiral_client/transpose/
  util/yclient)
- **APSI keyword:** keyword_pir_registrar, hint_gen, hint_serialize, lwe_params,
  hint_cache — **pass here, GFW-blocked on .50**

## Notes

- On `.50` the canonical green is **70 of 72 analysed** (the 2 APSI targets
  fail to *build* due to GFW, not a regression). The local box closes that gap.
- See `spiral-calibration-notes.md` for the SpiralPIR AVX2 correctness fix and
  the gitcode fork rationale.
