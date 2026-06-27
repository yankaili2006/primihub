# SpiralPIR "Is correct?: 0" — ROOT CAUSE FOUND (2026-06-27)

**Resolved by a controlled standalone build on a GPU/AVX512 box.** The bug is an
**upstream correctness bug in spiral's non-AVX512 fallback**, NOT a primihub
calibration/param issue. (An earlier revision of this note guessed a
defines↔(nu_1,nu_2) mismatch — that hypothesis was **disproved**; see below.)

## The finding (one line)

`menonsamir/spiral` only decodes correctly when compiled with **AVX512**
(`__AVX512F__`). Its scalar/AVX2 fallback in `multiplyQueryByDatabase`
(`src/spiral.cpp` `#else` at L932–997, vs the AVX512 `#if` at L640–931) is a
stale, **incorrect** reimplementation. `.50` is Broadwell (AVX2 only) →
`-march=native` defines no `__AVX512F__` → the broken fallback runs →
`Is correct?: 0`.

## Controlled experiment (same source, same SPIRAL_DEFINES, same HEXL .a)

Built upstream spiral standalone on the local RTX-5070-Ti box (HEXL 1.2.5 + the
cpu_features objects pulled from .50), `./spiral nu1 nu2 idx a --random-data`:

| build flags | `Is correct?` |
|---|---|
| `-O3 -march=native` (AVX512 present) | **1** for every nu: (8,2)(9,4)(10,10)(4,4)(5,5)(8,6) |
| `-O3 -march=native -mno-avx512f`     | **0** |
| `-O3 -march=broadwell`               | **0** |

The primihub `EnsureInitialized`+`SmokeTest` driving sequence, replicated
verbatim in a standalone `main`, also prints **1** under AVX512 — so the
in-process driving and params are correct. And building primihub's *actual*
`spiral_runtime_test` real-mode ON .50 (via `--override_repository=spiral_pir=`
my local source) reproduced **`Is correct?: 0`** — confirming it's the .50
(Broadwell) build, not the logic.

→ The discriminator is **exactly `__AVX512F__`**.

## Why the fallback is wrong (`src/spiral.cpp` L640 vs L932)

`multiplyQueryByDatabase` is the first-dimension matmul. The two branches are
different algorithms, not SIMD-width variants of one:

- **AVX512 (L640–931):** packed-CRT representation — two residues packed in a
  64-bit word, extracted via `>> packed_offset_2` + `_mm512_mul_epu32` (low-32×
  low-32), accumulating `sums_out_n0` / `sums_out_n2` with periodic
  `% q_intermediate`.
- **scalar `#else` (L932–997):** naive `lo = v_a`, `hi = v_a>>32` split,
  `sums_out_n0 += lo*b_lo`, `sums_out_n1 += hi*b_hi`, final `% p_i` / `% b_i`.
  This does **not** implement the packed_offset_2 / q_intermediate scheme, so
  the residues come out wrong. Independently, L937 also uses
  `rand() % dummyWorkingSet` where the AVX512 path (L647) uses
  `z % dummyWorkingSet` (a second divergence; fixing it alone does not help —
  verified).

## Fix options

1. **Recommended / pragmatic — require AVX512 for SpiralPIR.** The Spiral paper
   targets AVX512 servers; production / GPU hosts have it; on AVX512 the default
   primihub build already decodes correctly. Make `SpiralPirOperator` advertise
   AVX512 as required (e.g. `caps.is_real`/availability gated on
   `__builtin_cpu_supports("avx512f")`, or `backends={AVX512}` only), so it is
   not presented as correct on AVX2-only hosts like `.50` (a dev box). No upstream
   patching needed.
2. **Proper — port the scalar fallback.** Rewrite the L932–997 `#else` to mirror
   the AVX512 packed-CRT arithmetic exactly (same `packed_offset_2`,
   `q_intermediate`, `sums_out_n0`/`sums_out_n2` packing and reductions), via a
   WORKSPACE `patch_cmds` on the spiral pin. Multi-hour upstream-arithmetic work;
   only needed if SpiralPIR must run correctly on AVX2-only hardware.

## Reproduction recipe (standalone, no bazel)

```
# fetch src/*.cpp + include/*.h from menonsamir/spiral @361ee47f (raw.github…)
# HEXL: reuse a prebuilt libhexl.a (1.2.5) + cpu_features .o (e.g. from .50 cache)
DEFS="-DTEXP=8 -DTEXPRIGHT=56 -DTCONV=4 -DTGSW=10 -DQPBITS=22 -DPVALUE=256 \
      -DQNUMFIRST=1 -DQNUMREST=0 -DOUTN=2"
g++ -std=c++17 -O3 -march=native -fopenmp -include omp.h $DEFS -Iinclude -I<hexl> \
    src/{spiral,core,constants,poly,util,client,testing}.cpp libhexl.a cpuobj/*.o \
    -lpthread -o spiral
OMP_NUM_THREADS=1 ./spiral 5 5 17 a --random-data | grep 'Is correct'   # -> 1
# add -mno-avx512f to the same line  -> 0   (reproduces the bug)
```

## Evidence trail
- `src/spiral.cpp`: `multiplyQueryByDatabase` L628; AVX512 `#if` L640; packed
  arithmetic L687–708; scalar `#else` L932–997 (rand() L937, lo/hi split L955–975).
- primihub: `thirdparty/pir/BUILD.spiral` (`disable_avx512` → `-mno-avx512f`),
  `spiral_pir/spiral_runtime.cc` (EnsureInitialized/SmokeTest — correct).
