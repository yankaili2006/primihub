#!/usr/bin/env bash
# pir_runtime_activations.sh — Build and run the activation smoke tests
# for every PIR algorithm whose upstream is currently pinned + scaffolded
# (DoublePIR / SimplePIR / YPIR). Emits a single JSON report capturing
# per-algorithm pass/fail + wall time + binary sha256.
#
# This bench script is the CI-friendly entry point for verifying that
# the per-algorithm activation pattern still works after upstream pins
# move or after a refactor to the runtime facades:
#
#   * Each algorithm has a runtime facade (xxx_runtime.{h,cc}) that
#     either delegates to upstream C/C++ matmul kernels (real mode) or
#     returns retcode::FAIL with the activation-flag guidance (stub
#     mode).
#   * The smoke is "kernel link works + matmul math is identical to
#     in-line expected", NOT "full PIR algorithm works correctly" —
#     the algorithm cores stay open as multi-day Rust/Go-to-C++ ports.
#
# Usage:
#   bench/pir_runtime_activations.sh                  # default — all 3 algos
#   bench/pir_runtime_activations.sh --algos double_pir,simple_pir
#   bench/pir_runtime_activations.sh --out /tmp/out.json
#   bench/pir_runtime_activations.sh --no-build       # skip build, run existing
#
# Output JSON shape:
#   {
#     "schema_version": 1,
#     "captured_at": "2026-06-07T01:23:45Z",
#     "wrapper_commit": "<git HEAD on bench host>",
#     "host_uname": "<uname -srm>",
#     "cells": [
#       {
#         "algorithm": "double_pir",
#         "define_flag": "enable_double_pir_real",
#         "upstream_override_repo": "simplepir",
#         "upstream_override_path": "/tmp/simplepir-upstream",
#         "test_target": "//src/primihub/kernel/pir/tests:double_pir_runtime_test",
#         "binary_path": "bazel-bin/.../double_pir_runtime_test",
#         "binary_sha256": "...",
#         "build_exit_code": 0,
#         "test_exit_code": 0,
#         "wall_time_ms": 4123,
#         "smoke_log_line": "double_pir_runtime.cc:96] ...kernel link validated...",
#         "passed": true
#       },
#       ...
#     ],
#     "summary": {
#       "total": 3,
#       "passed": 3,
#       "failed": 0
#     }
#   }
#
# Exit codes:
#   0  — all activations pass
#   1  — at least one activation failed (build or test)
#   2  — bad arguments
set -euo pipefail

ALGOS_DEFAULT="double_pir,simple_pir,ypir"
ALGOS=""
OUT_PATH=""
DO_BUILD=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --algos)    ALGOS="$2"; shift 2;;
    --out)      OUT_PATH="$2"; shift 2;;
    --no-build) DO_BUILD=0; shift;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

ALGOS="${ALGOS:-$ALGOS_DEFAULT}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
mkdir -p "$REPO_ROOT/bench/results"

if [[ -z "$OUT_PATH" ]]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  OUT_PATH="$REPO_ROOT/bench/results/pir_runtime_activations_${TS}.json"
fi

# Per-algorithm wiring. Keep these in lockstep with the corresponding
# operator BUILD :enable_real config_setting + runtime header.
declare -A DEFINE_FLAG OVERRIDE_REPO OVERRIDE_PATH TEST_TARGET

DEFINE_FLAG[double_pir]="enable_double_pir_real"
OVERRIDE_REPO[double_pir]="simplepir"
OVERRIDE_PATH[double_pir]="/tmp/simplepir-upstream"
TEST_TARGET[double_pir]="//src/primihub/kernel/pir/tests:double_pir_runtime_test"

DEFINE_FLAG[simple_pir]="enable_simple_pir_real"
OVERRIDE_REPO[simple_pir]="simplepir"
OVERRIDE_PATH[simple_pir]="/tmp/simplepir-upstream"
TEST_TARGET[simple_pir]="//src/primihub/kernel/pir/tests:simple_pir_runtime_test"

DEFINE_FLAG[ypir]="enable_ypir_real"
OVERRIDE_REPO[ypir]="ypir"
OVERRIDE_PATH[ypir]="/tmp/ypir-upstream"
TEST_TARGET[ypir]="//src/primihub/kernel/pir/tests:ypir_runtime_test"

WRAPPER_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
HOST_UNAME=$(uname -srm)
CAPTURED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

CELLS=()
PASS=0
FAIL=0

for algo in ${ALGOS//,/ }; do
  if [[ -z "${DEFINE_FLAG[$algo]:-}" ]]; then
    echo "unknown algorithm: $algo" >&2
    exit 2
  fi

  flag="${DEFINE_FLAG[$algo]}"
  override_repo="${OVERRIDE_REPO[$algo]}"
  override_path="${OVERRIDE_PATH[$algo]}"
  target="${TEST_TARGET[$algo]}"

  echo "[bench] $algo  --define=${flag}=1  --override_repository=${override_repo}=${override_path}" >&2

  if [[ ! -d "$override_path" ]]; then
    echo "  ! override path missing: $override_path — skipping" >&2
    CELLS+=("$(python3 -c "import json; print(json.dumps({
      'algorithm': '$algo', 'define_flag': '$flag',
      'upstream_override_repo': '$override_repo',
      'upstream_override_path': '$override_path',
      'test_target': '$target',
      'build_exit_code': -1, 'test_exit_code': -1,
      'wall_time_ms': 0, 'binary_path': None, 'binary_sha256': None,
      'smoke_log_line': None,
      'passed': False, 'skip_reason': 'override path missing'}))")")
    FAIL=$((FAIL + 1))
    continue
  fi

  BUILD_RC=0
  if [[ "$DO_BUILD" -eq 1 ]]; then
    if ! bazel build --config=linux_x86_64 \
          --define="${flag}=1" \
          --override_repository="${override_repo}=${override_path}" \
          "$target" >&2; then
      BUILD_RC=$?
    fi
  fi

  TARGET_REL="${target#//}"
  TARGET_REL="${TARGET_REL/:/\/}"
  BINARY="$REPO_ROOT/bazel-bin/${TARGET_REL}"
  if [[ ! -x "$BINARY" ]]; then
    echo "  ! built binary not found at $BINARY" >&2
    CELLS+=("$(python3 -c "import json; print(json.dumps({
      'algorithm': '$algo', 'define_flag': '$flag',
      'upstream_override_repo': '$override_repo',
      'upstream_override_path': '$override_path',
      'test_target': '$target', 'binary_path': '$BINARY',
      'binary_sha256': None,
      'build_exit_code': $BUILD_RC, 'test_exit_code': -1,
      'wall_time_ms': 0, 'smoke_log_line': None,
      'passed': False, 'skip_reason': 'binary not found'}))")")
    FAIL=$((FAIL + 1))
    continue
  fi

  BINARY_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
  TMP_OUT=$(mktemp)
  T0=$(date +%s%N)
  if "$BINARY" --gtest_filter='*SmokeMatchesVendoredFlag' \
      > "$TMP_OUT" 2>&1; then
    TEST_RC=0
  else
    TEST_RC=$?
  fi
  T1=$(date +%s%N)
  WALL_MS=$(( (T1 - T0) / 1000000 ))

  SMOKE_LINE=$(grep -E 'kernel link validated' "$TMP_OUT" | head -1 || true)
  rm -f "$TMP_OUT"

  if [[ "$TEST_RC" -eq 0 ]]; then PASS=$((PASS + 1)); else FAIL=$((FAIL + 1)); fi

  CELLS+=("$(python3 -c "import json; print(json.dumps({
    'algorithm': '$algo', 'define_flag': '$flag',
    'upstream_override_repo': '$override_repo',
    'upstream_override_path': '$override_path',
    'test_target': '$target', 'binary_path': '$BINARY',
    'binary_sha256': '$BINARY_SHA',
    'build_exit_code': $BUILD_RC, 'test_exit_code': $TEST_RC,
    'wall_time_ms': $WALL_MS,
    'smoke_log_line': '''$SMOKE_LINE'''.strip() or None,
    'passed': $TEST_RC == 0}))")")
done

CELLS_JSON=$(printf ',%s' "${CELLS[@]}")
CELLS_JSON="[${CELLS_JSON:1}]"

python3 -c "
import json
out = {
  'schema_version': 1,
  'captured_at': '$CAPTURED_AT',
  'wrapper_commit': '$WRAPPER_COMMIT',
  'host_uname': '$HOST_UNAME',
  'cells': json.loads('''$CELLS_JSON'''),
  'summary': {'total': $PASS + $FAIL, 'passed': $PASS, 'failed': $FAIL}
}
with open('$OUT_PATH', 'w') as f:
  json.dump(out, f, indent=2)
print('wrote', '$OUT_PATH')
" >&2

echo "$OUT_PATH"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
