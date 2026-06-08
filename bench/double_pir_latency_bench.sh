#!/usr/bin/env bash
# double_pir_latency_bench.sh — sweep DoublePirOperator across a range of
# DB sizes and emit a JSON report capturing per-N setup + per-query
# latency. Mirrors the pir_runtime_activations.sh / pir_correctness_smoke.sh
# JSON shape so the trio composes into a single CI bundle.
#
# Usage:
#   bench/double_pir_latency_bench.sh                          # default sweep
#   bench/double_pir_latency_bench.sh --n-list "64 256 1024"   # custom list
#   bench/double_pir_latency_bench.sh --queries 16             # queries per N
#   bench/double_pir_latency_bench.sh --trials 3               # repeat per N
#   bench/double_pir_latency_bench.sh --out /tmp/out.json
#
# Default sweep: 64 / 256 / 1024 / 4096 (small enough to run in < 5s
# total). 1e6 / 1e8 are the long-term targets — runnable by passing
# --n-list "1000000 100000000" once a host with enough RAM is reachable
# (each entry is one byte, so N=1e8 is 100 MB raw; intermediate matrices
# grow to several GB during Setup).
#
# Each cell of the JSON output captures:
#   * setup_ms          — one-time Init+Setup cost
#   * per_query_ms      — avg query (Query+Answer+Recover) latency
#   * trials            — number of repetitions averaged
#   * binary_sha256     — fingerprint of the bench binary used
#
# The bench is gated behind PIR_BENCH_RUN_BAZEL=1 + SIMPLEPIR_UPSTREAM
# env var (same pattern as the correctness smoke) so a fresh checkout
# doesn't accidentally trigger a 30 s build on every CI invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_LIST="64 256 1024 4096"
QUERIES=16
TRIALS=1
OUT_PATH=""
BENCH_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-list)   N_LIST="$2";   shift 2 ;;
    --queries)  QUERIES="$2";  shift 2 ;;
    --trials)   TRIALS="$2";   shift 2 ;;
    --out)      OUT_PATH="$2"; shift 2 ;;
    --binary)   BENCH_BIN="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${PIR_BENCH_RUN_BAZEL:-}" ]]; then
  cat >&2 <<MSG
Bench requires building DoublePIR with vendored kernels. Re-invoke with:
  PIR_BENCH_RUN_BAZEL=1 SIMPLEPIR_UPSTREAM=<path to simplepir clone> $0 $*
MSG
  exit 0
fi
UPSTREAM="${SIMPLEPIR_UPSTREAM:-}"
if [[ -z "$UPSTREAM" || ! -d "$UPSTREAM" ]]; then
  echo "SIMPLEPIR_UPSTREAM=$UPSTREAM is not a directory" >&2
  exit 2
fi

# Find the bazel WORKSPACE root by walking up from the script.
ROOT="$SCRIPT_DIR"
while [[ "$ROOT" != "/" && ! -f "$ROOT/WORKSPACE" && ! -f "$ROOT/WORKSPACE.bazel" ]]; do
  ROOT="$(dirname "$ROOT")"
done
if [[ ! -f "$ROOT/WORKSPACE" && ! -f "$ROOT/WORKSPACE.bazel" ]]; then
  echo "WORKSPACE not found above $SCRIPT_DIR" >&2
  exit 2
fi

if [[ -z "$BENCH_BIN" ]]; then
  (cd "$ROOT" && bazel build --config=linux_x86_64 \
        --define=enable_pir_core_real=1 \
        --define=enable_double_pir_real=1 \
        --override_repository=simplepir="$UPSTREAM" \
        //src/primihub/kernel/pir/bench:double_pir_latency_bench >/dev/null)
  BENCH_BIN="$ROOT/bazel-bin/src/primihub/kernel/pir/bench/double_pir_latency_bench"
fi
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "bench binary not found / not executable: $BENCH_BIN" >&2
  exit 2
fi

CAPTURED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST_UNAME="$(uname -srm)"
BIN_SHA="$(sha256sum "$BENCH_BIN" | awk '{print $1}')"
WRAPPER_COMMIT="$(cd "$ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cells_path="$TMP/cells.json"
: > "$cells_path"
first=1
for N in $N_LIST; do
  setup_sum=0
  perq_sum=0
  trial_count=0
  status=ok
  for ((t = 0; t < TRIALS; ++t)); do
    line=$("$BENCH_BIN" --n "$N" --queries "$QUERIES" --csv 2>/dev/null || echo "$N,NA,NA,NA,binary_failed")
    # Parse "n,setup_ms,p50,pmax,status"
    IFS=, read -r got_n setup p50 _pmax tstatus <<<"$line"
    if [[ "$tstatus" != "ok" ]]; then
      status="$tstatus"
      break
    fi
    setup_sum=$(awk -v a="$setup_sum" -v b="$setup" 'BEGIN { print a + b }')
    perq_sum=$(awk -v a="$perq_sum" -v b="$p50"   'BEGIN { print a + b }')
    trial_count=$((trial_count + 1))
  done
  if [[ "$status" == "ok" && $trial_count -gt 0 ]]; then
    setup_avg=$(awk -v s="$setup_sum" -v t="$trial_count" 'BEGIN { printf "%.3f", s/t }')
    perq_avg=$(awk -v s="$perq_sum"  -v t="$trial_count" 'BEGIN { printf "%.3f", s/t }')
  else
    setup_avg=null
    perq_avg=null
  fi
  comma=""
  [[ $first -eq 0 ]] && comma=","
  first=0
  cat >> "$cells_path" <<JSON
${comma}{
  "n": $N,
  "queries": $QUERIES,
  "trials": $trial_count,
  "setup_ms": $setup_avg,
  "per_query_ms": $perq_avg,
  "status": "$status"
}
JSON
done

OUT_DEFAULT="$ROOT/bench/results/double_pir_latency_$(date -u +%Y%m%dT%H%M%SZ).json"
OUT_PATH="${OUT_PATH:-$OUT_DEFAULT}"
mkdir -p "$(dirname "$OUT_PATH")"
{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "captured_at": "%s",\n' "$CAPTURED_AT"
  printf '  "wrapper_commit": "%s",\n' "$WRAPPER_COMMIT"
  printf '  "host_uname": "%s",\n'    "$HOST_UNAME"
  printf '  "binary_path": "%s",\n'   "$BENCH_BIN"
  printf '  "binary_sha256": "%s",\n' "$BIN_SHA"
  printf '  "n_list": "%s",\n'        "$N_LIST"
  printf '  "queries": %d,\n'         "$QUERIES"
  printf '  "trials_per_n": %d,\n'    "$TRIALS"
  printf '  "cells": [\n'
  cat "$cells_path"
  printf '  ]\n'
  printf '}\n'
} > "$OUT_PATH"
echo "wrote $OUT_PATH"
