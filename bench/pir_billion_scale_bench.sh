#!/usr/bin/env bash
# pir_billion_scale_bench.sh — end-to-end client benchmark of the multi-algorithm
# PIR stack at 亿级 / billion scale (openspec change primihub-pir-cuda-tiptoe,
# task 3.1 / parent 11.4). Drives the NEW client with algorithm="spiral" /
# "double_pir" (/ "auto") against a DEPLOYED primihub node at N = 1e8 (default)
# or 1e9, sweeping algorithms x trials and recording the offline (setup/hint)
# cost + online per-query latency + throughput into a schema_version=1 JSON
# (same shape as bench/double_pir_latency_bench.sh so the results compose).
#
# This is the harness; the actual run needs a reachable deployed node with a DB
# of the target size loaded. WITHOUT --node it runs in dry-run mode: it validates
# the sweep, prints the exact client command it would issue per cell, writes a
# JSON of status="needs_deployed_node" cells, and exits 0 (so CI / a fresh
# checkout never blocks on infra). Point it at a node to get real numbers.
#
# Usage:
#   bench/pir_billion_scale_bench.sh                                  # dry-run plan
#   bench/pir_billion_scale_bench.sh --node host:50050 --n 100000000 \
#       --algorithm "spiral,double_pir" --queries 32 --trials 3 --out out.json
#
# Flags:
#   --node ADDR        deployed primihub node address (host:port). Omit = dry-run.
#   --n N              DB element count (default 100000000 = 1e8; 1000000000 = 1e9).
#   --algorithm LIST   comma-separated: spiral,double_pir,auto (default spiral,double_pir).
#   --queries Q        online queries timed per trial (default 32).
#   --trials T         trials averaged per (algorithm,N) cell (default 3).
#   --binary PATH      client binary (default: primihub-cli on PATH or bazel-bin).
#   --out PATH         results JSON (default bench/results/pir_billion_scale_<ts>.json).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NODE=""
N=100000000
ALGOS="spiral,double_pir"
QUERIES=32
TRIALS=3
BINARY=""
OUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)      NODE="$2"; shift 2 ;;
    --n)         N="$2"; shift 2 ;;
    --algorithm) ALGOS="$2"; shift 2 ;;
    --queries)   QUERIES="$2"; shift 2 ;;
    --trials)    TRIALS="$2"; shift 2 ;;
    --binary)    BINARY="$2"; shift 2 ;;
    --out)       OUT_PATH="$2"; shift 2 ;;
    -h|--help)   sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -z "$OUT_PATH" ]] && OUT_PATH="$SCRIPT_DIR/results/pir_billion_scale_$(date -u +%Y%m%dT%H%M%SZ).json"
mkdir -p "$(dirname "$OUT_PATH")"

# Resolve the client binary (only needed for a real run).
if [[ -z "$BINARY" ]]; then
  BINARY="$(command -v primihub-cli || true)"
  [[ -z "$BINARY" && -x "$ROOT/bazel-bin/src/primihub/cli/cli" ]] && BINARY="$ROOT/bazel-bin/src/primihub/cli/cli"
fi

COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
DRY=0
[[ -z "$NODE" ]] && DRY=1

echo "pir_billion_scale_bench: N=$N algorithms=[$ALGOS] queries=$QUERIES trials=$TRIALS"
[[ "$DRY" == 1 ]] && echo "  (DRY-RUN: no --node given; printing plan only)"

# Build the per-cell task config the client consumes. The PirTaskRequest gained
# `algorithm` + `latency_budget` + `preferred_backend` in the proto/CLI
# extension (parent task 6); a billion-scale run loads a DB of $N single-byte
# elements server-side and issues $QUERIES index lookups.
write_task_conf() {  # $1=algorithm $2=conf_path
  cat > "$2" <<JSON
{
  "task": "PIR",
  "algorithm": "$1",
  "node": "$NODE",
  "db_size": $N,
  "element_bytes": 1,
  "queries": $QUERIES,
  "preferred_backend": "auto"
}
JSON
}

# THE ONE DEPLOYMENT-DEPENDENT POINT. Runs the client for one (algorithm) cell
# against $NODE and echoes "setup_ms per_query_ms" on stdout. The exact client
# invocation/flags must be confirmed against the deployed build; the default
# below uses primihub-cli + the generated task conf and wall-clocks the run.
run_one_cell() {  # $1=algorithm  -> prints "<setup_ms> <per_query_ms>"
  local algo="$1" conf; conf="$(mktemp)"
  write_task_conf "$algo" "$conf"
  # TODO(deploy): confirm the client subcommand/flags for a PIR task run.
  #   "$BINARY" --config "$conf" --report-timing
  # For now wall-clock the whole invocation and split offline vs online if the
  # client reports it; otherwise attribute all to setup.
  local t0 t1 ms
  t0="$(date +%s.%N)"
  "$BINARY" --config "$conf" >/dev/null 2>&1 || { rm -f "$conf"; echo "ERR ERR"; return; }
  t1="$(date +%s.%N)"
  ms="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a)*1000}')"
  rm -f "$conf"
  # Without per-phase timing from the client, report wall as setup and
  # per-query as wall/queries (refine once the client emits phase timings).
  awk -v w="$ms" -v q="$QUERIES" 'BEGIN{printf "%.3f %.3f", w, w/q}'
}

cells=""
IFS=',' read -ra ALGO_ARR <<< "$ALGOS"
for algo in "${ALGO_ARR[@]}"; do
  if [[ "$DRY" == 1 ]]; then
    conf="$(mktemp)"; write_task_conf "$algo" "$conf"
    echo "  [$algo] would run: $([[ -n "$BINARY" ]] && echo "$BINARY" || echo "primihub-cli") --config <conf> against <node>"
    echo "    task conf:"; sed 's/^/      /' "$conf"; rm -f "$conf"
    cell=$(printf '{"algorithm":"%s","n":%s,"queries":%s,"trials":%s,"setup_ms":null,"per_query_ms":null,"status":"needs_deployed_node","reason":"pass --node host:port with a %s-element DB loaded"}' "$algo" "$N" "$QUERIES" "$TRIALS" "$N")
  else
    [[ -z "$BINARY" ]] && { echo "client binary not found; pass --binary" >&2; exit 1; }
    sum_setup=0; sum_q=0; ok=1
    for ((t=1; t<=TRIALS; t++)); do
      read -r s q <<< "$(run_one_cell "$algo")"
      [[ "$s" == "ERR" ]] && { ok=0; break; }
      sum_setup="$(awk -v a="$sum_setup" -v b="$s" 'BEGIN{print a+b}')"
      sum_q="$(awk -v a="$sum_q" -v b="$q" 'BEGIN{print a+b}')"
    done
    if [[ "$ok" == 1 ]]; then
      as="$(awk -v s="$sum_setup" -v t="$TRIALS" 'BEGIN{printf "%.3f", s/t}')"
      aq="$(awk -v s="$sum_q" -v t="$TRIALS" 'BEGIN{printf "%.3f", s/t}')"
      qps="$(awk -v q="$aq" 'BEGIN{printf "%.2f", q>0?1000/q:0}')"
      echo "  [$algo] setup=${as}ms per_query=${aq}ms (${qps} qps)"
      cell=$(printf '{"algorithm":"%s","n":%s,"queries":%s,"trials":%s,"setup_ms":%s,"per_query_ms":%s,"throughput_qps":%s,"status":"ok"}' "$algo" "$N" "$QUERIES" "$TRIALS" "$as" "$aq" "$qps")
    else
      echo "  [$algo] FAILED"
      cell=$(printf '{"algorithm":"%s","n":%s,"status":"error"}' "$algo" "$N")
    fi
  fi
  cells="${cells:+$cells,}$cell"
done

cat > "$OUT_PATH" <<JSON
{
  "schema_version": 1,
  "bench": "pir_billion_scale",
  "captured_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "wrapper_commit": "$COMMIT",
  "host_uname": "$(uname -srm)",
  "node": "${NODE:-none}",
  "n": $N,
  "queries": $QUERIES,
  "trials_per_cell": $TRIALS,
  "dry_run": $([ "$DRY" == 1 ] && echo true || echo false),
  "cells": [$cells]
}
JSON
echo "wrote $OUT_PATH"
