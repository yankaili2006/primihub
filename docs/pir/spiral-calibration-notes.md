# SpiralPIR "Is correct?: 0" — root-cause diagnosis (2026-06-27)

Status: **root cause identified; fix requires a buildable host to verify**
(`.50` cannot build spiral real-mode — the `menonsamir/spiral` clone is
GFW-blocked there, same as APSI/SEAL).

## Symptom

`SpiralRuntime::SmokeTest` drives upstream `load_db()` + `do_test()` in-process.
`do_test()` (upstream `src/spiral.cpp` L2408) is upstream's *own* self-test:
generate_setup_and_query → process_crtd_query → (modswitch) → `check_final`,
which prints `Is correct?: <is_eq(pt_real, M_result)>` (L1494). We get `0`.

## What is NOT the cause (ruled out by reading upstream main())

Our `EnsureInitialized` + `SmokeTest` faithfully mirror upstream `main()`
(L1228–1344):
- start sequence (`omp_set_num_threads(1)`, `build_table()`, `scratch` malloc,
  `ntt_qprime = new NTT(2048, arb_qprime)`) — **matches**.
- `IDX_DIM0 = IDX_TARGET / (1<<further_dims)` — **matches**.
- `dummyWorkingSet = min((1<<25)/total_n, poly_len)` — **matches**.
- `do_MatPol_test()` — we skip it, but it's a side-effect-free NTT round-trip
  self-test (L1181) → harmless.
- `max_trials` — `do_test()` uses a **local** `num_trials = 1` (L2433), so the
  global is a no-op.
- `checking_for_debug` — `do_test()` sets it `true` itself (L2434).

So the in-process *driving* is correct. The fault is in the **parameters**.

## Root cause: compile-time defines and runtime (nu_1, nu_2) are decoupled

Upstream selects the crypto parameters as **one matched bundle** via
`select_params.py`:
- `param_f` (L337) emits exactly our define names:
  `TEXP TEXPRIGHT TCONV TGSW QPBITS PVALUE QNUMFIRST QNUMREST OUTN`.
- the optimized param vector (`all_ks`, L288) is
  `['p','q_prime_bits','nu_1','nu_2','t_GSW','t_conv','t_exp','t_exp_right']`
  — i.e. **`nu_1`/`nu_2` are co-optimized with the gadget dims + modulus**, and
  `pred()` only accepts `(nu_1,nu_2)` present in `exp_lut`/`fdim_lut`.
- upstream `CMakeLists.txt` substitutes `-D$(TEXP)…` from that single run.

primihub decoupled the two halves:
1. `thirdparty/pir/BUILD.spiral` **hardcodes one define set** —
   `TEXP=8 TEXPRIGHT=56 TCONV=4 TGSW=10 QPBITS=22 PVALUE=256 QNUMFIRST=1
   QNUMREST=0 OUTN=2` — for the "wiki" 1M-record / 256-byte config
   (query_size 14336 B), whose matched dims are `nu_1+nu_2 = 20`.
2. `spiral_pir.cc` then calls `EstimateParams(db_size = index+1, 256)` which
   picks `(nu_1,nu_2)` **independently** from the tiny SmokeTest DB
   (e.g. index small → `(4,4)`, `total_n=256`; the runtime test's 1024 →
   `(5,5)`).

The compiled crypto instance (modulus chain / gadget decomposition baked in by
`setup_constants()` from the 1M-config defines) is therefore run with
`nu_1/nu_2` it was not generated for → `check_final` decodes garbage → `0`.

Secondary suspicion to check during the fix: `QNUMFIRST=1, QNUMREST=0` is a
single-prime (~2^28) modulus; verify it actually carries the 1M config's
`nu_2`-fold noise budget — it may be an incomplete copy of the wiki define set.

## Fix (needs verification on a buildable host)

Make the runtime dims match the compiled defines. Two options:

- **Quick (v1 / SmokeTest correctness):** pin `EnsureInitialized` to the exact
  `(nu_1, nu_2)` that `select_params.py 20 256` pairs with the hardcoded
  defines, instead of using `EstimateParams`'s independent split. Then
  `do_test` should print `Is correct?: 1`.
- **Proper (multi-size support):** run `select_params.py <logN> 256` per target
  size to generate *both* the `-D` define set *and* the matched `(nu_1,nu_2)`,
  and have the build select the define set + `EstimateParams` return the paired
  dims (a small generated table keyed by logN). This is exactly what upstream's
  build flow does.

## Reproduction recipe (on a non-GFW host with a CUDA-free toolchain)

```
git clone https://github.com/menonsamir/spiral && cd spiral   # commit 361ee47f
# get the precomputed optimizer tables the script needs:
#   exp_lut.json, fdim_lut.json, all_params.pkl  (in-repo / via build-*-lut)
python3 select_params.py 20 256 --show-output --dry-run   # -> defines + (nu_1,nu_2)
# confirm the hardcoded BUILD.spiral defines == this output, and note the dims.
# then in primihub:
bazel test --config=linux_x86_64 --define=enable_spiral_real=1 \
  --override_repository=spiral_pir=<clone> --override_repository=hexl=<hexl> \
  //src/primihub/kernel/pir/tests:spiral_runtime_test --test_output=all
# iterate until check_final prints "Is correct?: 1".
```

## Evidence trail
- upstream `src/spiral.cpp`: main() L1228–1344, do_test() L2408, check_final()
  L1412 (verdict L1494), load_db() L1028 (pt_real planted at IDX_TARGET L1132).
- upstream `select_params.py`: `param_f` L337, `all_ks` L288, `pred()` L305.
- primihub: `thirdparty/pir/BUILD.spiral` SPIRAL_DEFINES; `spiral_pir/params.h`
  `EstimateParams` (kMinNu=4, balanced split); `spiral_pir/spiral_runtime.cc`
  EnsureInitialized + SmokeTest.
