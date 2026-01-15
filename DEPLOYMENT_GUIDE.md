# PrimiHub 本地编译部署指南

## 构建状态
✅ **构建成功完成**

已成功编译以下组件：
- `primihub-node` - 主节点程序
- `primihub-cli` - 命令行工具

## 快速启动

### 1. 启动节点
```bash
# 启动节点0
./primihub-node --node_id=node0 --config=config/primihub_node0.yaml

# 启动节点1 (新终端)
./primihub-node --node_id=node1 --config=config/primihub_node1.yaml

# 启动节点2 (新终端)  
./primihub-node --node_id=node2 --config=config/primihub_node2.yaml
```

### 2. 测试隐私求交任务
```bash
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
```

### 3. 查看结果
```bash
cat data/result/psi_result.csv
```

## 配置文件
- `config/primihub_node0.yaml` - 节点0配置
- `config/primihub_node1.yaml` - 节点1配置  
- `config/primihub_node2.yaml` - 节点2配置

## 数据目录
- `data/` - 测试数据文件
- `data/result/` - 任务结果输出目录

## 注意事项
- 确保所有节点配置文件中的端口不冲突
- 首次运行可能需要创建必要的目录结构
- 如需MySQL支持，需要安装 `libmysqlclient-dev` 包