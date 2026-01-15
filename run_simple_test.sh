#!/bin/bash

echo "=== 启动PrimiHub服务 ==="
./start_server.sh

echo -e "\n等待服务启动..."
sleep 5

echo -e "\n=== 检查服务状态 ==="
ps aux | grep -E "(node|fusion)" | grep -v grep

echo -e "\n=== 测试Python环境 ==="
source venv/bin/activate
export PYTHONPATH=/home/primihub/github/primihub/python:$PYTHONPATH

python3 -c "
import pandas as pd
print('Python环境正常')
print('Pandas版本:', pd.__version__)

# 读取示例数据
df = pd.read_csv('data/test_party_0.csv')
print(f'测试数据: {df.shape[0]}行, {df.shape[1]}列')
print('数据预览:')
print(df.head(3))
"

echo -e "\n=== 测试CLI ==="
echo "CLI版本:"
./primihub-cli --version

echo -e "\n=== 测试完成 ==="
echo "服务正在运行中..."
echo "要停止服务，运行: ./stop_server.sh"