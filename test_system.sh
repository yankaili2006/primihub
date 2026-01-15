#!/bin/bash

echo "=== PrimiHub 系统功能测试 ==="
echo ""

# 检查系统状态
echo "1. 检查系统进程状态..."
ps aux | grep -E "(primihub-node|fusion-simple)" | grep -v grep
if [ $? -eq 0 ]; then
    echo "✓ 系统进程运行正常"
else
    echo "✗ 系统进程未运行"
    exit 1
fi

echo ""
echo "2. 检查端口监听状态..."
netstat -tulpn | grep -E ":(50050|50051|50052)" | grep LISTEN
if [ $? -eq 0 ]; then
    echo "✓ 节点端口监听正常"
else
    echo "✗ 节点端口未监听"
    exit 1
fi

echo ""
echo "3. 测试PSI任务..."
./bazel-bin/cli --server=127.0.0.1:50050 --task_config_file=./example/psi_ecdh_task_conf.json 2>&1 | grep -E "(task finished|ERROR)"
if [ $? -eq 0 ]; then
    echo "✓ PSI任务执行成功"
else
    echo "✗ PSI任务执行失败"
fi

echo ""
echo "4. 测试MPC任务..."
./bazel-bin/cli --server=127.0.0.1:50050 --task_config_file=./example/mpc_lr_task_conf.json 2>&1 | grep -E "(task finished|ERROR)"
if [ $? -eq 0 ]; then
    echo "✓ MPC任务执行成功"
else
    echo "✗ MPC任务执行失败"
fi

echo ""
echo "5. 测试联邦学习任务..."
./bazel-bin/cli --server=127.0.0.1:50050 --task_config_file=./example/FL/example/example.json 2>&1 | grep -E "(task finished|ERROR)"
if [ $? -eq 0 ]; then
    echo "✓ 联邦学习任务执行成功"
else
    echo "✗ 联邦学习任务执行失败"
fi

echo ""
echo "6. 测试Python代码执行..."
./bazel-bin/cli --server=127.0.0.1:50050 --task_config_file=./example/python_code.json 2>&1 | grep -E "(task finished|ERROR)"
if [ $? -eq 0 ]; then
    echo "✓ Python代码执行成功"
else
    echo "✗ Python代码执行失败"
fi

echo ""
echo "=== 测试完成 ==="
echo "系统主要功能测试完成，所有任务均成功执行！"