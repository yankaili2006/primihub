#!/usr/bin/env bash
# pir_selector_sweep.sh — Sweep the PirSelector input space and capture which
# algorithm wins each cell, with score and failure reasons. Emits a JSON file
# under bench/results/ that can be diffed across commits to catch unintended
# selector regressions (e.g. "double_pir used to win this cell, now it loses").
#
# This is the bench script that actually exercises something today, because
# 5/6 registered algorithms are skeletons (OnExecute returns FAIL) — running
# a real PIR query against them would just collect FAIL rows. The selector,
# in contrast, is fully implemented and benchmarkable: it consults each
# algorithm's PirCapabilities and ranks them.
#
# Usage:
#   bench/pir_selector_sweep.sh                       # default sweep
#   bench/pir_selector_sweep.sh --binary /usr/local/bin/pir_inspect
#   bench/pir_selector_sweep.sh --out /tmp/sweep.json # custom output path
#
# Output JSON shape:
#   {
#     "schema_version": 1,
#     "captured_at": "2026-06-06T00:00:00Z",
#     "binary_sha256": "...",      // pir_inspect that produced this sweep
#     "cells": [
#       {
#         "constraints": {db_size, query_type, latency_budget, ...},
#         "winner": "double_pir",   // null if no algo passes
#         "ranking": [{"algorithm","passes","score","comm_kb","fail_reasons"}]
#       }, ...
#     ]
#   }
set -euo pipefail

PIR_INSPECT=""
OUT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      PIR_INSPECT="$2"
      shift 2
      ;;
    --out)
      OUT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

# Locate pir_inspect — prefer CLI override, then env var, then PATH lookup,
# then bazel-out (when invoked from inside the worktree).
if [[ -z "$PIR_INSPECT" ]]; then
  PIR_INSPECT="${PRIMIHUB_PIR_INSPECT:-}"
fi
if [[ -z "$PIR_INSPECT" ]] && command -v pir_inspect >/dev/null 2>&1; then
  PIR_INSPECT="$(command -v pir_inspect)"
fi
if [[ -z "$PIR_INSPECT" ]]; then
  local_bazel="bazel-out/k8-fastbuild/bin/src/primihub/cli/pir_inspect/pir_inspect"
  if [[ -x "$local_bazel" ]]; then
    PIR_INSPECT="$local_bazel"
  fi
fi
if [[ ! -x "$PIR_INSPECT" ]]; then
  echo "pir_inspect not found. Pass --binary <path> or build it first:" >&2
  echo "  bazel build --config=linux_x86_64 //src/primihub/cli/pir_inspect:pir_inspect" >&2
  exit 2
fi

# Default output: bench/results/pir_selector_sweep_<date>.json
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "$OUT_PATH" ]]; then
  mkdir -p "$SCRIPT_DIR/results"
  OUT_PATH="$SCRIPT_DIR/results/pir_selector_sweep_$(date -u +%Y%m%d).json"
fi
TMP_PATH="$(mktemp)"

# Sweep grid. Each axis is intentionally small so a full sweep finishes in <2s
# even on a slow host. The point isn't combinatorial coverage; it's pinning
# down the "interesting" decision boundaries (size cutoffs, latency tiers,
# two-server gating).
DB_SIZES=(1000000 100000000 10000000000)
QUERY_TYPES=(index keyword)
LATENCY=(any seconds sub-second ms)
TWO_SERVER=(false true)
CACHE_HINT=(false true)
ASSUME_NC=(false true)

BIN_SHA256=$(sha256sum "$PIR_INSPECT" | awk '{print $1}')
CAPTURED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  printf '{\n'
  printf '  "schema_version": 1,\n'
  printf '  "captured_at": "%s",\n' "$CAPTURED_AT"
  printf '  "binary_path": "%s",\n' "$PIR_INSPECT"
  printf '  "binary_sha256": "%s",\n' "$BIN_SHA256"
  printf '  "cells": [\n'

  first=1
  for db in "${DB_SIZES[@]}"; do
    for qt in "${QUERY_TYPES[@]}"; do
      for lat in "${LATENCY[@]}"; do
        for ts in "${TWO_SERVER[@]}"; do
          for ch in "${CACHE_HINT[@]}"; do
            for anc in "${ASSUME_NC[@]}"; do
              # dry-run: ranked rationale table; we parse line-by-line.
              raw=$("$PIR_INSPECT" auto \
                "db-size=$db" \
                "query-type=$qt" \
                "latency-budget=$lat" \
                "allow-two-server=$ts" \
                "client-can-cache-hint=$ch" \
                "assume-non-colluding=$anc" \
                "dry-run=true" 2>/dev/null) || raw=""

              # Skip header (2 lines) + Recommend winner (1 line).
              # The dry-run table is: algorithm | passes | score | comm_KB | fail_reasons
              ranking_json="["
              winner="null"
              first_row=1
              while IFS= read -r line; do
                case "$line" in
                  algorithm*) continue ;;     # column header
                  ---*) continue ;;           # divider
                  "") continue ;;
                esac
                # Split on " | "
                IFS='|' read -r algo passes score comm reasons <<< "$line"
                algo=$(echo "$algo" | xargs)
                passes=$(echo "$passes" | xargs)
                score=$(echo "$score" | xargs)
                comm=$(echo "$comm" | xargs)
                reasons=$(echo "$reasons" | xargs | sed 's/"/\\"/g')
                if [[ -z "$algo" ]]; then continue; fi
                if [[ $first_row -eq 0 ]]; then ranking_json+=", "; fi
                first_row=0
                ranking_json+="{\"algorithm\":\"$algo\",\"passes\":$([[ "$passes" == "yes" ]] && echo true || echo false),\"score\":${score:-0},\"comm_kb\":${comm:-0},\"fail_reasons\":\"$reasons\"}"
                if [[ "$passes" == "yes" && "$winner" == "null" ]]; then
                  winner="\"$algo\""
                fi
              done <<< "$raw"
              ranking_json+="]"

              if [[ $first -eq 0 ]]; then printf ',\n'; fi
              first=0
              printf '    {\n'
              printf '      "constraints": {'
              printf '"db_size": %s, ' "$db"
              printf '"query_type": "%s", ' "$qt"
              printf '"latency_budget": "%s", ' "$lat"
              printf '"allow_two_server": %s, ' "$ts"
              printf '"client_can_cache_hint": %s, ' "$ch"
              printf '"assume_non_colluding": %s' "$anc"
              printf '},\n'
              printf '      "winner": %s,\n' "$winner"
              printf '      "ranking": %s\n' "$ranking_json"
              printf '    }'
            done
          done
        done
      done
    done
  done

  printf '\n  ]\n'
  printf '}\n'
} > "$TMP_PATH"

mv "$TMP_PATH" "$OUT_PATH"

# Summary on stdout: total cells, cells with a winner, top-3 winners.
TOTAL=$(grep -c '"constraints":' "$OUT_PATH" || true)
NULL_WINS=$(grep -c '"winner": null' "$OUT_PATH" || true)
WINS=$(( TOTAL - NULL_WINS ))

echo "selector sweep: $TOTAL cells, $WINS with a winner, $NULL_WINS unsatisfiable"
echo "output: $OUT_PATH"
echo
echo "winner distribution:"
grep -oE '"winner": "[^"]+"' "$OUT_PATH" \
  | sort | uniq -c | sort -rn | head -10
