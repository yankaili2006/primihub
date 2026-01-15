#!/bin/bash

echo "=== PrimiHub 实际任务测试 ==="
echo "当前时间: $(date)"
echo

# 检查服务状态
echo "1. 检查服务状态..."
SERVICE_COUNT=$(ps aux | grep -E "(node|fusion)" | grep -v grep | wc -l)
if [ $SERVICE_COUNT -ge 6 ]; then
    echo "✓ 服务正在运行 ($SERVICE_COUNT 个进程)"
else
    echo "✗ 服务未完全启动，正在启动服务..."
    ./start_server.sh
    sleep 10
fi

# 检查端口
echo -e "\n2. 检查端口监听..."
PORTS="50050 50051 50052 7977 7978 7979"
for port in $PORTS; do
    if netstat -tln | grep -q ":$port "; then
        echo "✓ 端口 $port 正在监听"
    else
        echo "✗ 端口 $port 未监听"
    fi
done

# 测试PSI任务
echo -e "\n3. 测试PSI隐私求交任务..."
if [ -f "example/psi_ecdh_task_conf.json" ]; then
    echo "运行PSI ECDH任务..."
    timeout 30 ./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
    PSI_RESULT=$?
    
    if [ $PSI_RESULT -eq 0 ]; then
        echo "✓ PSI任务执行成功"
        
        # 检查结果文件
        if [ -f "data/result/psi_result.csv" ]; then
            echo "PSI结果文件内容:"
            head -5 "data/result/psi_result.csv"
            LINE_COUNT=$(wc -l < "data/result/psi_result.csv")
            echo "结果行数: $LINE_COUNT"
        else
            echo "✗ PSI结果文件未生成"
        fi
    else
        echo "✗ PSI任务执行失败 (退出码: $PSI_RESULT)"
    fi
else
    echo "✗ PSI任务配置文件不存在"
fi

# 测试MPC统计任务
echo -e "\n4. 测试MPC统计求和任务..."
if [ -f "example/mpc_statistics_sum_task_conf.json" ]; then
    echo "运行MPC统计求和任务..."
    timeout 30 ./primihub-cli --task_config_file="example/mpc_statistics_sum_task_conf.json"
    MPC_RESULT=$?
    
    if [ $MPC_RESULT -eq 0 ]; then
        echo "✓ MPC统计任务执行成功"
    else
        echo "✗ MPC统计任务执行失败 (退出码: $MPC_RESULT)"
    fi
else
    echo "✗ MPC统计任务配置文件不存在"
fi

# 测试MPC算术任务
echo -e "\n5. 测试MPC算术任务..."
MPC_ARITH_TASKS=(
    "example/mpc_add_task_conf.json"
    "example/mpc_sub_task_conf.json"
    "example/mpc_mul_task_conf.json"
    "example/mpc_div_task_conf.json"
)

for task_file in "${MPC_ARITH_TASKS[@]}"; do
    if [ -f "$task_file" ]; then
        task_name=$(basename "$task_file")
        echo "测试 $task_name..."
        timeout 20 ./primihub-cli --task_config_file="$task_file" >/dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "  ✓ $task_name 执行成功"
        else
            echo "  ✗ $task_name 执行失败"
        fi
    fi
done

# 检查Python示例
echo -e "\n6. 测试Python示例..."
if [ -f "example/code/example.py" ]; then
    echo "运行Python示例..."
    source venv/bin/activate
    export PYTHONPATH=/home/primihub/github/primihub/python:$PYTHONPATH
    timeout 10 python3 "example/code/example.py" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✓ Python示例执行成功"
    else
        echo "✗ Python示例执行失败"
    fi
else
    echo "Python示例文件不存在"
fi

# 汇总结果
echo -e "\n=== 测试结果汇总 ==="
echo "服务状态: $(if [ $SERVICE_COUNT -ge 6 ]; then echo "正常"; else echo "异常"; fi)"
echo "PSI测试: $(if [ $PSI_RESULT -eq 0 ]; then echo "成功"; else echo "失败"; fi)"
echo "MPC统计测试: $(if [ $MPC_RESULT -eq 0 ]; then echo "成功"; else echo "失败"; fi)"
echo "Python示例: $(if [ -f "example/code/example.py" ] && timeout 5 python3 -c "print('check')" >/dev/null 2>&1; then echo "可用"; else echo "不可用"; fi)"

echo -e "\n=== 下一步建议 ==="
echo "1. 查看详细日志: tail -f log_node0"
echo "2. 运行更多测试: ./primihub-cli --task_config_file=\"example/mpc_lr_task_conf.json\""
echo "3. 查看数据结果: ls -la data/result/"
echo "4. 停止服务: ./stop_server.sh"
echo "5. 清理结果: rm -rf data/result/*"