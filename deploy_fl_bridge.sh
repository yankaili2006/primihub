#!/bin/bash
# Deploy FL Python bridge scripts to application containers
# This allows FL tasks to execute without Web UI interaction
SCRIPT_DIR=$(dirname "$0")
for app in application0 application1 application2; do
  docker exec $app mkdir -p /home/primihub/primihub-platform/python-algorithms/federated_learning 2>/dev/null
  for script in logistic_regression_train linear_regression_train xgboost_train; do
    docker cp "$SCRIPT_DIR/fl_bridge_script.py" $app:/home/primihub/primihub-platform/python-algorithms/federated_learning/${script}.py 2>/dev/null
  done
  echo "$app: FL bridge deployed"
done
