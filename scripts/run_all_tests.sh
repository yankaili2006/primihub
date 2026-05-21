#!/bin/bash
# Run all PrimiHub tests
set -e
cd "$(dirname "$0")/.."
START_TIME=$(date +%s)

echo "=========================================="
echo " PrimiHub Test Suite"
echo "=========================================="

echo ""
echo "=== 1. C++ Bazel Tests ==="
echo ""

C_TARGETS=(
    "//test/primihub/util:util_func_test"
    "//test/primihub/util:arrow_wrapper_test"
    "//test/primihub/util:file_util_test"
    "//test/primihub/util/network:mock_channel_test"
    "//test/primihub/common/config:config_decode_test"
    "//test/primihub/common/type:common_test"
    "//test/primihub/cli:task_config_parser_test"
    "//test/primihub/data_store:csv_driver_test"
    "//test/primihub/data_store:parquet_driver_test"
    "//test/primihub/data_store:sqlite_driver_test"
    "//test/primihub/kernel/psi:psi_util_test"
    "//test/primihub/kernel/psi:base_psi_test"
)

for target in "${C_TARGETS[@]}"; do
    echo "  Building $target..."
    bazel build --config=linux_x86_64 --jobs=4 "$target" -q 2>/dev/null
done

echo ""
echo "  Running C++ tests..."
PASS=0
FAIL=0
for target in "${C_TARGETS[@]}"; do
    name=$(echo "$target" | sed 's|//test/primihub/||;s|:|/|')
    bin="bazel-bin/test/primihub/$name"
    if [ -f "$bin" ]; then
        if $bin 2>/dev/null | grep -q "FAILED"; then
            echo "  ✗ $name"
            FAIL=$((FAIL + 1))
        else
            echo "  ✓ $name"
            PASS=$((PASS + 1))
        fi
    fi
done
echo "  C++: $PASS passed, $FAIL failed"

echo ""
echo "=== 2. Python Tests ==="
echo ""
./scripts/coverage_python.sh 2>/dev/null | tail -5

END_TIME=$(date +%s)
echo ""
echo "=========================================="
echo " Total time: $((END_TIME - START_TIME))s"
echo "=========================================="
