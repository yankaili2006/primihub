#!/usr/bin/env bash
# frodo_pir_latency_bench.sh — sweep FrodoPirOperator across a range of
# DB sizes and emit a JSON report capturing per-N setup + per-query
# latency. Sibling of double_pir_latency_bench.sh / simple_pir_persistence_bench.sh.
#
# Usage:
#   bench/frodo_pir_latency_bench.sh                          # default sweep
#   bench/frodo_pir_latency_bench.sh --n-list "1000 10000"    # custom list
#   bench/frodo_pir_latency_bench.sh --queries 8              # queries per N
#   bench/frodo_pir_latency_bench.sh --out /tmp/out.json
#
# Default sweep: 1000 / 10000 / 100000 (small enough to run in < 1 min
# total on .50 worktree). 1e6 / 1e7 are the long-term targets — runnable
# by passing --n-list "1000000 10000000" once a host has the wall-clock
# budget. 1e8 is impractical with the current unoptimized port (~30 min
# Setup + ~3 min per_query).
#
# FrodoPIR is self-contained — no SIMPLEPIR_UPSTREAM dependency, no
# --define flags. The bench builds in the default config.
#
# Each cell of the JSON output captures:
#   * setup_ms          — one-time Shard::FromBase64Strings cost
#   * per_query_ms      — avg QueryParams::New + GenerateQuery + Respond
#                          + ParseOutputAsBase64
#   * wall_ms           — total bench wall-clock
#   * queries           — queries per cell
#   * binary_sha256     — fingerprint of the bench binary used

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_LIST="1000 10000 100000"
QUERIES=8
OUT_PATH=""
BENCH_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-list)   N_LIST="$2";   shift 2 ;;
    --queries)  QUERIES="$2";  shift 2 ;;
    --out)      OUT_PATH="$2"; shift 2 ;;
    --binary)   BENCH_BIN="$2"; shift 2 ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# //'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BENCH_BIN" ]]; then
  local_bazel="bazel-out/k8-opt/bin/src/primihub/kernel/pir/bench/frodo_pir_latency_bench"
  if [[ -x "$local_bazel" ]]; then
    BENCH_BIN="$local_bazel"
  else
    local_bazel="bazel-bin/src/primihub/kernel/pir/bench/frodo_pir_latency_bench"
    if [[ -x "$local_bazel" ]]; then
      BENCH_BIN="$local_bazel"
    fi
  fi
fi
if [[ -z "$BENCH_BIN" || ! -x "$BENCH_BIN" ]]; then
  cat >&2 <<MSG
frodo_pir_latency_bench binary not found. Build it first:
  bazel build --config=linux_x86_64 -c opt \\
    //src/primihub/kernel/pir/bench:frodo_pir_latency_bench
MSG
  exit 2
fi

if [[ -z "$OUT_PATH" ]]; then
  mkdir -p "$SCRIPT_DIR/results"
  OUT_PATH="$SCRIPT_DIR/results/pir_frodo_latency_$(date -u +%Y%m%d).json"
fi
TMP_PATH="$(mktemp)"

CAPTURED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
BIN_SHA="$(sha256sum "$BENCH_BIN" | awk '{print $1}')"

{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "captured_at": "%s",\n' "$CAPTURED_AT"
  printf '  "binary_path": "%s",\n' "$BENCH_BIN"
  printf '  "binary_sha256": "%s",\n' "$BIN_SHA"
  printf '  "algorithm": "frodo_pir",\n'
  printf '  "lwe_dim": 512,\n'
  printf '  "plaintext_bits": 10,\n'
  printf '  "queries_per_cell": %s,\n' "$QUERIES"
  printf '  "results": [\n'

  first=1
  for n in $N_LIST; do
    [[ $first -eq 1 ]] || printf ',\n'
    first=0
    line=$("$BENCH_BIN" --n "$n" --queries "$QUERIES" --csv) || true
    # CSV: n,setup_ms,per_query_ms,wall_ms,status
    n_field=$(echo "$line" | awk -F',' '{print $1}')
    setup_ms=$(echo "$line" | awk -F',' '{print $2}')
    per_query_ms=$(echo "$line" | awk -F',' '{print $3}')
    wall_ms=$(echo "$line" | awk -F',' '{print $4}')
    status=$(echo "$line" | awk -F',' '{print $5}')
    printf '    {"n": %s, "setup_ms": %s, "per_query_ms": %s, "wall_ms": %s, "status": "%s"}' \
      "$n_field" "$setup_ms" "$per_query_ms" "$wall_ms" "$status"
  done
  printf '\n  ]\n'
  printf '}\n'
} > "$TMP_PATH"

mv "$TMP_PATH" "$OUT_PATH"

echo "frodo_pir latency bench:"
n_count=$(echo "$N_LIST" | wc -w)
echo "  cells captured: $n_count"
echo "  binary sha256:  $BIN_SHA"
echo "  output:         $OUT_PATH"
