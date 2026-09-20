#!/usr/bin/env bash
# simple_pir_persistence_bench.sh — quantifies the SimpleHintCache
# cold-start vs warm-start setup_ms gap (SimplePIR sibling of
# bench/double_pir_persistence_bench.sh, task 5.6 chunks 1-5 mirror).
#
# Same shape as the DoublePIR wrapper: per N, runs cold (rm -f the
# hint file first) then warm (same hint file), emits per-cell JSON
# with a wall_speedup ratio.
#
# Usage:
#   PIR_BENCH_RUN_BAZEL=1 SIMPLEPIR_UPSTREAM=/tmp/simplepir-upstream \
#     bench/simple_pir_persistence_bench.sh \
#       --n-list "64 1024 4096" --queries 4 --output bench/results/sp_persist.json

set -euo pipefail

WORKTREE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N_LIST="64 256 1024 4096"
QUERIES=4
OUTPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-list) N_LIST="$2"; shift 2;;
    --queries) QUERIES="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    -h|--help) grep '^#' "$0" | head -25; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ "${PIR_BENCH_RUN_BAZEL:-0}" != "1" ]]; then
  echo "skip: PIR_BENCH_RUN_BAZEL not set" >&2
  exit 0
fi
if [[ -z "${SIMPLEPIR_UPSTREAM:-}" ]]; then
  echo "error: SIMPLEPIR_UPSTREAM must point to a local simplepir clone" >&2
  exit 2
fi

cd "$WORKTREE_ROOT"
TARGET="//src/primihub/kernel/pir/bench:simple_pir_latency_bench"
echo "building $TARGET..." >&2
bazel build --config=linux_x86_64 --define=enable_pir_core_real=1 \
  --override_repository="simplepir=$SIMPLEPIR_UPSTREAM" "$TARGET" >&2

BIN="bazel-bin/src/primihub/kernel/pir/bench/simple_pir_latency_bench"
HINT_FILE=$(mktemp -t simple_pir_persist_XXXXXX)
rm -f "$HINT_FILE"

trap 'rm -f "$HINT_FILE"' EXIT

run_cell() {
  local n="$1"
  "$BIN" --n "$n" --queries "$QUERIES" --hint-path "$HINT_FILE" --csv 2>/dev/null
}

results=()
for n in $N_LIST; do
  rm -f "$HINT_FILE"
  cold=$(run_cell "$n")
  warm=$(run_cell "$n")
  cold_init=$(awk -F, '{print $2}' <<< "$cold")
  cold_setup=$(awk -F, '{print $3}' <<< "$cold")
  cold_pq=$(awk -F, '{print $4}' <<< "$cold")
  cold_wall=$(awk -F, '{print $5}' <<< "$cold")
  cold_hit=$(awk -F, '{print $6}' <<< "$cold")
  warm_init=$(awk -F, '{print $2}' <<< "$warm")
  warm_setup=$(awk -F, '{print $3}' <<< "$warm")
  warm_pq=$(awk -F, '{print $4}' <<< "$warm")
  warm_wall=$(awk -F, '{print $5}' <<< "$warm")
  warm_hit=$(awk -F, '{print $6}' <<< "$warm")
  speedup=$(awk -v c="$cold_wall" -v w="$warm_wall" 'BEGIN{
    if (w+0 == 0) printf "null"; else printf "%.3f", c/w
  }')
  results+=("{\"n\":$n,\"queries\":$QUERIES,\"cold\":{\"init_ms\":$cold_init,\"setup_ms\":$cold_setup,\"per_query_ms\":$cold_pq,\"wall_ms\":$cold_wall,\"hint_hit\":$cold_hit},\"warm\":{\"init_ms\":$warm_init,\"setup_ms\":$warm_setup,\"per_query_ms\":$warm_pq,\"wall_ms\":$warm_wall,\"hint_hit\":$warm_hit},\"wall_speedup\":$speedup}")
done

joined=""
for r in "${results[@]}"; do
  joined="${joined:+$joined,}$r"
done
DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOST=$(hostname)
SCHEMA=1
BAZEL_BIN_SHA=$(sha256sum "$BIN" | awk '{print $1}')
JSON="{\"schema_version\":$SCHEMA,\"algorithm\":\"simple_pir\",\"timestamp\":\"$DATE\",\"host\":\"$HOST\",\"binary_sha256\":\"$BAZEL_BIN_SHA\",\"queries\":$QUERIES,\"results\":[$joined]}"

if [[ -n "$OUTPUT" ]]; then
  mkdir -p "$(dirname "$OUTPUT")"
  echo "$JSON" > "$OUTPUT"
  echo "wrote $OUTPUT" >&2
else
  echo "$JSON"
fi
