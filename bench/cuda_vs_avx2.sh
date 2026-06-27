#!/usr/bin/env bash
# Three-backend PIR Answer benchmark: scalar CPU vs AVX2 CPU vs CUDA GPU, on the
# LWE matrix-vector product answer = A*q mod 2^32 at DB ~1e8 (openspec change
# primihub-pir-cuda-tiptoe, task 2.4).
#
# Needs nvcc + a GPU for the CUDA path (e.g. the local RTX 5070 Ti box). On a
# CPU-only host (e.g. .50, no GPU) it prints a notice and exits 0 -- the CUDA
# backend simply isn't measurable there.
#
# Usage: bench/cuda_vs_avx2.sh [output.json]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/cuda_vs_avx2_bench.cu"
OUT_JSON="${1:-}"

NVCC="$(command -v nvcc || true)"
[ -z "$NVCC" ] && [ -x /usr/local/cuda/bin/nvcc ] && NVCC=/usr/local/cuda/bin/nvcc
if [ -z "$NVCC" ]; then
  echo "[cuda_vs_avx2] nvcc not found -- CPU-only host (e.g. .50). Skipping."
  echo "[cuda_vs_avx2] Run on a GPU host with the CUDA toolkit for backend numbers."
  exit 0
fi

BIN="$(mktemp -d)/cuda_vs_avx2_bench"
echo "[cuda_vs_avx2] building with $("$NVCC" --version | grep -o 'release [0-9.]*')"
"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"

echo "[cuda_vs_avx2] running:"
LOG="$(mktemp)"
"$BIN" | tee "$LOG"

# Optional machine-readable snapshot (parsed from the table).
if [ -n "$OUT_JSON" ]; then
  gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
  awk -v gpu="$gpu" '
    /^  scalar/        {sc=$2}
    /^  avx2/          {av=$2}
    /^  cuda .warm/    {cw=$3}
    /^  cuda .cold/    {cc=$3}
    /DB upload one-time/ {for(i=1;i<=NF;i++) if(($i+0)==$i && $i!="") up=$i}
    END {
      printf "{\"op\":\"lwe_matvec_mod2p32\",\"db_entries\":1e8,\"gpu\":\"%s\",", gpu
      printf "\"per_answer_ms\":{\"scalar\":%s,\"avx2\":%s,\"cuda_warm\":%s,\"cuda_cold\":%s},", sc, av, cw, cc
      printf "\"db_upload_ms\":%s}\n", up
    }' "$LOG" > "$OUT_JSON"
  echo "[cuda_vs_avx2] wrote $OUT_JSON"
fi
