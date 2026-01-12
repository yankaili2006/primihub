#!/bin/bash

echo "🚀 启动 PrimiHub 本地集群..."

# 检查必要的目录
mkdir -p log localdb

# 启动元数据服务 (需要Java环境)
echo "📊 启动元数据服务..."
if command -v java >/dev/null 2>&1; then
    # 启动元数据服务
    cd meta_service/meta_service
    java -jar meta-server.jar \
        --server.port=8088 \
        --grpc.server.port=9099 \
        --db.path=../../localdb/meta_service/node0 &
    META_PID=$!
    cd ../..
    echo "元数据服务启动 (PID: $META_PID)"
    sleep 5
else
    echo "⚠️  未找到Java环境，跳过元数据服务启动"
    echo "   节点将以独立模式运行"
fi

# 启动节点
echo "🖥️  启动节点..."

# 节点0
./primihub-node --node_id=node0 --config=config/primihub_node0_local.yaml > log/node0.log 2>&1 &
NODE0_PID=$!
echo "节点0启动 (PID: $NODE0_PID)"

# 节点1  
sleep 2
./primihub-node --node_id=node1 --config=config/primihub_node1_local.yaml > log/node1.log 2>&1 &
NODE1_PID=$!
echo "节点1启动 (PID: $NODE1_PID)"

# 节点2
sleep 2  
./primihub-node --node_id=node2 --config=config/primihub_node2_local.yaml > log/node2.log 2>&1 &
NODE2_PID=$!
echo "节点2启动 (PID: $NODE2_PID)"

echo ""
echo "✅ PrimiHub 集群启动完成!"
echo ""
echo "📝 进程信息:"
echo "   节点0: PID $NODE0_PID (日志: log/node0.log)"
echo "   节点1: PID $NODE1_PID (日志: log/node1.log)"  
echo "   节点2: PID $NODE2_PID (日志: log/node2.log)"
if [ ! -z "$META_PID" ]; then
    echo "   元数据服务: PID $META_PID"
fi
echo ""
echo "🔧 测试任务:"
echo "   ./primihub-cli --task_config_file=\"example/psi_ecdh_task_conf.json\""
echo ""
echo "🛑 停止集群:"
echo "   pkill primihub-node"
if [ ! -z "$META_PID" ]; then
    echo "   kill $META_PID"
fi

# 保存PID文件
echo $NODE0_PID > .node0.pid
echo $NODE1_PID > .node1.pid
echo $NODE2_PID > .node2.pid
if [ ! -z "$META_PID" ]; then
    echo $META_PID > .meta.pid
fi