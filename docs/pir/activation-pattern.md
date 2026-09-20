# PIR Operator Activation Pattern

> Status: **3 operators activated** (DoublePIR / SimplePIR / YPIR) as of
> 2026-06-07. Pattern is the contract every future PIR operator port
> (FrodoPIR, Tiptoe, additional algorithms) is expected to follow.

PIR algorithm research code lives upstream in Go and Rust, not C++. primihub
intentionally avoids `rules_go` / `rules_rust` to keep the build closure
hermetic (see `thirdparty/pir/BUILD.simplepir` for the full rejected-paths
analysis). The result is a layered integration where each algorithm has:

1. **An upstream pin** in `WORKSPACE_GITHUB` with a primihub-maintained
   wrapper at `thirdparty/pir/BUILD.<algo>` that exposes only the C / C++
   matrix kernels (when the upstream has any) — not the algorithm core.
2. **A primihub C++ operator** at
   `src/primihub/kernel/pir/operator/<algo>/<algo>.{h,cc}` that ports the
   algorithmic layer to C++ on top of those kernels.

The **activation pattern** documented here is the boundary between (1)
and (2). It provides a definitive runtime signal that the cc_library
link works end-to-end before any algorithm-level code is written, so the
multi-day port work that follows builds on solid infrastructure.

---

## The four moving parts

| File | Role |
| --- | --- |
| `<algo>_runtime.h` | Public facade — declares `kXxxRuntimeVendored` constexpr + a `XxxRuntime` singleton with one or more smoke methods. No upstream types cross this header. |
| `<algo>_runtime.cc` | Two compile modes selected by `#ifdef PIR_<ALGO>_RUNTIME_VENDORED`. Vendored mode forward-declares the upstream `extern "C"` kernels and calls them. Stub mode returns `retcode::FAIL` with a populated `err` string that names the activation flag. |
| `<algo>/BUILD` | Adds a `config_setting(name="enable_real", define_values={"enable_<algo>_real": "1"})`, then a `select()` that injects both the `PIR_<ALGO>_RUNTIME_VENDORED` define AND the `@<upstream>//:<kernel_target>` dep when the flag is set. |
| `tests/<algo>_runtime_test.cc` | Two GTests: one that bifurcates on `kXxxRuntimeVendored` (requires SUCCESS in vendored mode, FAIL with activation-flag hint in stub mode), and one that asserts the smoke is idempotent (catches accidental statefulness sneaking into the runtime). |

The operator's own `OnExecute` then:

1. Checks `kXxxRuntimeVendored` — returns FAIL with the activation-flag
   guidance if false.
2. Calls `XxxRuntime::Instance().Smoke<Kernel>(&err)`.
3. On smoke FAIL, returns FAIL.
4. On smoke SUCCESS, **still returns FAIL** with a `WARNING` log noting
   that the full algorithm port is pending. This keeps callers from
   assuming a real PIR query happened. `kIsSkeleton` on the operator
   stays `true` until step (5).
5. (FUTURE) Once the algorithm is ported, replace the smoke call with
   the real query path and flip `kIsSkeleton` to `false`.

---

## Naming convention

Replace `<algo>` with the registered algorithm name (lowercased,
underscored). Three live examples as of 2026-06-07:

| Algorithm | define flag | runtime header | upstream kernel target |
| --- | --- | --- | --- |
| `double_pir` | `--define=enable_double_pir_real=1` | `double_pir_runtime.h` | `@simplepir//:simplepir_c_kernels` |
| `simple_pir` | `--define=enable_simple_pir_real=1` | `simple_pir_runtime.h` | `@simplepir//:simplepir_c_kernels` |
| `ypir` | `--define=enable_ypir_real=1` | `ypir_runtime.h` | `@ypir//:ypir_matmul_kernels` |

`double_pir` and `simple_pir` share the same upstream because the
ahenzinger/simplepir repository ships both algorithms in one Go module
with a single C kernel layer.

---

## Picking the smoke shape

The smoke method exists to **validate the kernel link**, not to validate
the algorithm. Three constraints shape its design:

1. **Tiny input.** A 3x3 matrix or 8x1 packed vector is enough. Larger
   inputs slow CI for no extra signal.
2. **In-line expected values.** Compute the expected output in the same
   `.cc` file the smoke lives in so the assertion is a single source of
   truth. Do not rely on test-vector files or upstream documentation
   (they go stale).
3. **Cover the actual link.** Call the specific upstream symbol the
   future algorithm port will depend on most. For `double_pir` and
   `simple_pir` that is `matMul` / `matMulVec`; for `ypir` that is the
   scalar `matMulVecPacked` (not the SIMD variants — those depend on
   host CPU capability and would make the smoke flaky in CI).

Two live examples illustrate the trade-off:

* `SimplePirRuntime::SmokeMatMul`: 3x3 identity matrix → output equals
  input. Tightest possible signal.
* `YpirRuntime::SmokeMatMulVecPacked`: 8x1 packed input with each byte
  set to 1 → each row contributes 4 (= COMPRESSION). Validates the
  packing semantics on top of the link.

---

## Adding a fourth algorithm — checklist

1. Decide whether the upstream has a C/C++ kernel surface. If not (e.g.
   FrodoPIR is 100% Rust, Tiptoe is 100% Go with no C), the wrapper at
   `thirdparty/pir/BUILD.<algo>` should be doc-only and this activation
   pattern does NOT apply — go straight to a full Rust/Go-to-C++ port
   inside the operator. Document the lack of a kernel surface in both
   `BUILD.<algo>` and the operator's `OnExecute` log message.
2. If there IS a kernel surface:
   * Write `<algo>_runtime.h` modeled on `simple_pir_runtime.h`. Pick
     ONE upstream symbol to forward-declare.
   * Write `<algo>_runtime.cc` modeled on `simple_pir_runtime.cc`. Two
     `#ifdef PIR_<ALGO>_RUNTIME_VENDORED` blocks: the extern "C" forward
     decl at file scope, then the smoke method body.
   * Edit `<algo>/BUILD` to add the `:enable_real` config_setting, the
     `defines` select(), and the `deps` select() that pulls in the
     upstream `@<repo>//:<kernel_target>`.
   * Edit `<algo>/<algo>.cc` `OnExecute` to call
     `<algo>::kXxxRuntimeVendored` + `XxxRuntime::Instance().Smoke...()`
     + return FAIL with WARNING.
   * Add `tests/<algo>_runtime_test.cc` with the two-test pattern.
3. Wire the algorithm into `bench/pir_runtime_activations.sh`:
   * Add a row to the `DEFINE_FLAG` / `OVERRIDE_REPO` / `OVERRIDE_PATH`
     / `TEST_TARGET` arrays at the top of the script.
   * Verify locally with
     `bench/pir_runtime_activations.sh --algos <algo>`.

---

## Cross-references

* Per-algorithm runtime headers:
  `src/primihub/kernel/pir/operator/{double_pir,simple_pir,ypir}/`
  `*_runtime.h`.
* Bench script: `bench/pir_runtime_activations.sh`. Run with `--no-build`
  for fast iteration; binary sha256s and per-cell `smoke_log_line` make
  the JSON output reproducible.
* Reference activation commits: `dc037df7` (DoublePIR — first one),
  `7fcd3e16` (SimplePIR + YPIR — second pair establishing the pattern).
* SpiralPIR (`spiral_pir_runtime.{h,cc}`) predates this pattern (commits
  548d1c48 / 9e43ee6d). It is the same shape but with a more elaborate
  smoke (full SmokeTest pipeline including `is_correct` invariant) —
  use it as a reference for adding a fuller smoke once an algorithm has
  a richer C++ kernel surface than matrix multiplication.
