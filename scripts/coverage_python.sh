#!/bin/bash
# Python coverage runner for PrimiHub local modules
set -e
cd "$(dirname "$0")/.."
VENV_PYTHON="./venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual env not found at $VENV_PYTHON"
    echo "Run: python3 -m venv venv && ./venv/bin/pip install pytest coverage"
    exit 1
fi
$VENV_PYTHON -m pip install coverage -q 2>/dev/null
$VENV_PYTHON -m coverage run --source=python/primihub/local \
    -m pytest python/primihub/tests/ -q \
    --ignore=python/primihub/tests/test_opt_paillier_c2py.py \
    --ignore=python/primihub/tests/test_opt_paillier_pack_c2py.py \
    --ignore=python/primihub/tests/express_test.py \
    --ignore=python/primihub/tests/test_dataset_warpper.py \
    --ignore=python/primihub/tests/test_dataset_register.py \
    --ignore=python/primihub/tests/utils_test/ \
    --ignore=python/primihub/tests/dataset/ 2>/dev/null
$VENV_PYTHON -m coverage report -m
