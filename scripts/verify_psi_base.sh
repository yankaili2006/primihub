#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

command -v bazel >/dev/null 2>&1 || {
  echo "bazel is required; run this on the configured Linux build VM" >&2
  exit 127
}

cd "$REPO_ROOT"
bazel test --config=linux_x86_64 \
  //test/primihub/kernel/psi:psi_util_test \
  //test/primihub/kernel/psi:ecdh_psi_mock_test \
  //test/primihub/kernel/psi:base_psi_test \
  --test_output=errors
