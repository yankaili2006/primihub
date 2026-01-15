#!/bin/bash

echo "🛑 停止 PrimiHub 本地集群..."

# 停止节点
if [ -f .node0.pid ]; then
    NODE0_PID=$(cat .node0.pid)
    kill $NODE0_PID 2>/dev/null && echo "✅ 停止节点0 (PID: $NODE0_PID)" || echo "⚠️  节点0未运行"
    rm -f .node0.pid
fi

if [ -f .node1.pid ]; then
    NODE1_PID=$(cat .node1.pid)
    kill $NODE1_PID 2>/dev/null && echo "✅ 停止节点1 (PID: $NODE1_PID)" || echo "⚠️  节点1未运行"
    rm -f .node1.pid
fi

if [ -f .node2.pid ]; then
    NODE2_PID=$(cat .node2.pid)
    kill $NODE2_PID 2>/dev/null && echo "✅ 停止节点2 (PID: $NODE2_PID)" || echo "⚠️  节点2未运行"
    rm -f .node2.pid
fi

# 停止元数据服务
if [ -f .meta.pid ]; then
    META_PID=$(cat .meta.pid)
    kill $META_PID 2>/dev/null && echo "✅ 停止元数据服务 (PID: $META_PID)" || echo "⚠️  元数据服务未运行"
    rm -f .meta.pid
fi

# 清理残留进程
pkill -f primihub-node 2>/dev/null && echo "🧹 清理残留节点进程"

sleep 2
echo ""
echo "✅ PrimiHub 集群已停止"