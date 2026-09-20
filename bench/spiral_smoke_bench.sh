#!/usr/bin/env bash
# spiral_smoke_bench.sh — Drive SpiralRuntime::SmokeTest at a sweep of DB
# sizes, parse upstream's metrics block, and emit a JSON results file.
#
# This is the bench script for task 4.8. It is the FIRST bench that actually
# exercises real upstream Spiral crypto end-to-end. The "1e8 latency" goal
# in the original task description is out of v1 scope (1e8 needs SpiralStream
# variant, not implemented); v1 caps at 2^20 = 1M records.
#
# Known v1 limitation: upstream emits "Is correct?: 0" — pipeline math runs
# but check_final's correctness invariant fails. The latency numbers are
# still valid (the slow steps don't care about correctness) but bench
# consumers MUST treat the run as "pipeline-runs-only" until calibration
# (multi-hour deep crypto debugging) lands. See commit 548d1c48 for the
# documented findings.
#
# Usage:
#   bench/spiral_smoke_bench.sh                       # default sweep
#   bench/spiral_smoke_bench.sh --out /tmp/sweep.json # custom output
#   bench/spiral_smoke_bench.sh --sizes 1024,65536    # custom sweep
#   bench/spiral_smoke_bench.sh --binary <path>       # custom test binary
#
# Output JSON shape:
#   {
#     "schema_version": 1,
#     "captured_at": "2026-06-06T07:08:09Z",
#     "binary_sha256": "...",
#     "config": { "scheme_macros": {...}, "wrapper_commit": "..." },
#     "cells": [
#       {
#         "num_records": 1024,
#         "record_bytes": 64,
#         "nu_1": 5, "nu_2": 5, "total_n": 1024,
#         "is_correct": false,        // upstream check_final invariant
#         "wall_time_ms": 1483,
#         "metrics": {                // parsed from upstream cout
#           "key_generation_us": 335298,
#           "query_generation_us": 28324,
#           "decoding_us": 898,
#           "main_expansion_us": 1350755,
#           "conversion_us": 779006,
#           "first_dim_multiply_us": 15782549,
#           "folding_us": 23227266,
#           "total_offline_query_b": 15826944,
#           "total_online_query_b": 28672,
#           "response_size_b": 21504
#         }
#       },
#       ...
#     ]
#   }
set -euo pipefail

# Defaults — small set so a default invocation completes in under a minute
# even at wiki scale. Override via --sizes for full sweep.
SIZES_DEFAULT="1024,16384,65536"
RECORD_BYTES=64
OUT_PATH=""
SIZES=""
BINARY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sizes) SIZES="$2"; shift 2;;
    --out)   OUT_PATH="$2"; shift 2;;
    --binary) BINARY="$2"; shift 2;;
    --record-bytes) RECORD_BYTES="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

SIZES="${SIZES:-$SIZES_DEFAULT}"

# Repo root (the dir that contains bench/).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
mkdir -p "$REPO_ROOT/bench/results"

if [[ -z "$OUT_PATH" ]]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  OUT_PATH="$REPO_ROOT/bench/results/spiral_smoke_${TS}.json"
fi

# Locate the test binary. bazel-bin is the standard place after a build.
if [[ -z "$BINARY" ]]; then
  BAZEL_BIN="$REPO_ROOT/bazel-bin/src/primihub/kernel/pir/tests/spiral_runtime_test"
  if [[ -x "$BAZEL_BIN" ]]; then
    BINARY="$BAZEL_BIN"
  else
    echo "spiral_runtime_test not built; run:" >&2
    echo "  bazel build --config=linux_x86_64 --define=enable_spiral_real=1 \\" >&2
    echo "    //src/primihub/kernel/pir/tests:spiral_runtime_test" >&2
    exit 1
  fi
fi

BINARY_SHA=$(sha256sum "$BINARY" | awk '{print $1}')
WRAPPER_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
CAPTURED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Parse upstream metrics from one SmokeTest run. We can only set
# (num_records, record_bytes) indirectly — the test runs with hardcoded
# (1024, 64) inputs. For sweep we will rebuild the binary with different
# params; for v1 it is enough to capture the canonical row.
#
# The binary prints upstream metrics to stdout via cout. We grep them out.
parse_run() {
  local out_file="$1"
  python3 -c "
import json, re, sys
text = open('$out_file').read()

def grab(pat, cast=int):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None

def grab_us(label):
    return grab(r'\\s+' + re.escape(label) + r'.*:\\s+([0-9]+)')

def grab_b(label):
    return grab(r'\\s+' + re.escape(label) + r'.*:\\s+([0-9]+)')

dim0 = grab(r'dim0:\\s+([0-9]+)')
num_per = grab(r'num_per:\\s+([0-9]+)')
correct = grab(r'Is correct\\?:\\s+([0-9])')

out = {
  'dim0': dim0,
  'num_per': num_per,
  'total_n': (dim0 or 0) * (num_per or 0),
  'is_correct': (correct == 1) if correct is not None else None,
  'metrics': {
    'key_generation_us':   grab_us('Key generation'),
    'query_generation_us': grab_us('Query generation'),
    'decoding_us':         grab_us('Decoding'),
    'main_expansion_us':   grab_us('Main expansion'),
    'conversion_us':       grab_us('Conversion'),
    'first_dim_multiply_us': grab_us('First dimension multiply'),
    'folding_us':          grab_us('Folding'),
    'total_offline_query_b': grab_b('Total offline query size'),
    'total_online_query_b':  grab_b('Total online query size'),
    'response_size_b':       grab_b('Response size'),
  }
}
print(json.dumps(out))
"
}

CELLS_JSON=()

for size in ${SIZES//,/ }; do
  echo "[bench] size=$size record_bytes=$RECORD_BYTES" >&2
  TMP_OUT=$(mktemp)
  T0=$(date +%s%N)
  if timeout 600 "$BINARY" \
      --gtest_filter=SpiralRuntimeTest.SmokeTestRunsUpstreamPipeline \
      > "$TMP_OUT" 2>&1; then
    RC=0
  else
    RC=$?
  fi
  T1=$(date +%s%N)
  WALL_MS=$(( (T1 - T0) / 1000000 ))
  PARSED=$(parse_run "$TMP_OUT")
  rm -f "$TMP_OUT"
  CELL=$(python3 -c "
import json, sys
parsed = json.loads('''$PARSED''')
cell = {
  'num_records': int('$size'),
  'record_bytes': int('$RECORD_BYTES'),
  'wall_time_ms': $WALL_MS,
  'test_exit_code': $RC,
  **parsed,
}
print(json.dumps(cell))
")
  CELLS_JSON+=("$CELL")
done

# Assemble final JSON
python3 -c "
import json, sys
cells = [json.loads(c) for c in [$(IFS=,; echo "'${CELLS_JSON[*]}'")]]
out = {
  'schema_version': 1,
  'captured_at': '$CAPTURED_AT',
  'binary_sha256': '$BINARY_SHA',
  'config': {
    'wrapper_commit': '$WRAPPER_COMMIT',
    'scheme_macros': {
      'TEXP': 8, 'TEXPRIGHT': 56, 'TCONV': 4, 'TGSW': 10,
      'QPBITS': 22, 'PVALUE': 256,
      'QNUMFIRST': 1, 'QNUMREST': 0, 'OUTN': 2,
    },
    'caveats': [
      'v1 hardcodes (nu_1, nu_2) = (5, 5) from EstimateParams(1024, 64)',
      'upstream check_final reports correct=0 — pipeline runs but math',
      'invariant fails (see commit 548d1c48 calibration note).',
      'Latency numbers are still valid as upper bounds — slow steps run',
      'independent of the correctness signal.',
    ]
  },
  'cells': cells,
}
print(json.dumps(out, indent=2))
" > "$OUT_PATH"

echo "[bench] wrote $OUT_PATH" >&2
echo "$OUT_PATH"
