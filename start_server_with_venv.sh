#!/bin/bash
set -x

# 激活虚拟环境
source venv/bin/activate

# 设置Python路径
export PYTHONPATH=/home/primihub/github/primihub/python:$PYTHONPATH

# 添加C++扩展模块路径
export PYTHONPATH=/home/primihub/github/primihub/bazel-bin/src/primihub/pybind_warpper:$PYTHONPATH
export PYTHONPATH=/home/primihub/github/primihub/bazel-bin/src/primihub/task/pybind_wrapper:$PYTHONPATH

# 运行原始启动脚本
./start_server.sh