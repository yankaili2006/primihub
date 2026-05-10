#!/bin/bash
# PrimiHub C++ coverage runner
set -e
cd "$(dirname "$0")/.."
echo "=== Building and running tests with coverage ==="
bazel coverage --config=linux_x86_64 --jobs=4 \
    //test/primihub/util:util_func_test \
    //test/primihub/util:arrow_wrapper_test \
    //test/primihub/common/config:config_decode_test \
    //test/primihub/cli:task_config_parser_test \
    //test/primihub/kernel/psi:psi_util_test \
    --combined_report=lcov 2>/dev/null
echo "=== Coverage report generated ==="
ls -la coverage_report.lcov 2>/dev/null && echo "Done." || echo "No coverage report generated"
echo ""
echo "Python coverage:"
./scripts/coverage_python.sh 2>&1 | tail -3
