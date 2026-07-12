# PrimiHub 部署文档

## 1. 环境要求

### 1.1 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|---------|---------|
| CPU | 4 核 | 8 核以上 |
| 内存 | 8 GB | 32 GB |
| 磁盘 | 50 GB | 200 GB+ SSD |
| 网络 | 100 Mbps | 1 Gbps |

### 1.2 软件要求

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Docker | 20.10+ | 容器化部署（推荐） |
| Docker Compose | 2.0+ | 多容器编排 |
| 操作系统 | Ubuntu 20.04+ / CentOS 8+ | 支持 x86_64 和 ARM64 |
| Git | 2.0+ | 下载源码 |
| Bazel | 5.0+ | 从源码编译（可选） |
| gRPC | 1.42+ | 通信协议 |

### 1.3 网络要求

| 端口 | 协议 | 用途 | 是否必须开放 |
|------|------|------|------------|
| 50050-50052 | gRPC/TCP | 节点间通信 | 是 |
| 9099 | gRPC/TCP | 元数据服务 | 是 |
| 8088 | HTTP/TCP | 元数据健康检查 | 建议 |
| 6666 | gRPC/TCP | 通知服务 | 按需 |
| 6379 | TCP | Redis（可选） | 按需 |

---

## 2. 快速部署（Docker）

### 2.1 一键部署

```bash
# 1. 克隆代码仓库
git clone https://github.com/primihub/primihub.git
cd primihub

# 2. 启动所有服务（3 计算节点 + 3 元数据服务）
docker-compose up -d

# 3. 验证部署状态
docker-compose ps

# 4. 查看运行日志
docker-compose logs -f
```

### 2.2 验证部署

```bash
# 检查容器运行状态（应看到 6 个 running 容器）
docker ps

# 进入节点容器
docker exec -it primihub-node0 bash

# 执行 PSI 任务验证
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"

# 查看任务结果
cat data/result/psi_result.csv
```

### 2.3 自定义镜像

```bash
# 使用自定义镜像仓库
REGISTRY=registry.cn-beijing.aliyuncs.com/primihub TAG=v1.0 docker-compose up -d

# 使用指定版本
TAG=latest docker-compose up -d
```

---

## 3. 从源码编译部署

### 3.1 环境准备

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-dev \
  libssl-dev libgmp-dev libntl-dev libboost-all-dev

# CentOS/RHEL
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake git python3-devel openssl-devel \
  gmp-devel ntl-devel boost-devel
```

### 3.2 安装 Bazel

```bash
# 下载 Bazel 5.0.0
wget https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-installer-linux-x86_64.sh
chmod +x bazel-5.0.0-installer-linux-x86_64.sh
sudo ./bazel-5.0.0-installer-linux-x86_64.sh

# 验证安装
bazel version
```

### 3.3 编译项目

```bash
# 1. 安装系统依赖
bash pre_build.sh

# 2. 编译（使用 Make 封装）
make mysql=y

# 或直接使用 Bazel
bazel build //:node //:cli //:task_main

# 3. 验证编译产物
ls -la bazel-bin/
# 应包含: node, cli, task_main
```

### 3.4 编译选项

```bash
make release              # 发布模式（优化 + 去除调试符号）
make debug=y              # 调试模式（启用 ASAN）
make mysql=y              # 启用 MySQL 数据源支持
make tee=y                # 启用 Intel SGX TEE 支持
make jobs=4               # 指定并行编译线程数

# 完整发布编译
make release mysql=y jobs=8

# 最小化编译（仅基础功能）
bazel build //:node //:cli --config=opt
```

### 3.5 编译常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Bazel 下载依赖超时 | 网络问题 | 使用 `WORKSPACE_CN` 国内镜像 |
| Python 头文件找不到 | Python-dev 未安装 | `apt install python3-dev` |
| gRPC 编译失败 | Protobuf 版本冲突 | 清除缓存: `bazel clean --expunge` |
| 内存不足 | Bazel 编译消耗大 | 限制并行: `--jobs=4` |
| GCC 版本过低 | C++17 需要 GCC 9+ | `apt install gcc-9 g++-9` |

---

## 4. 本地集群部署

### 4.1 单机多节点部署

```bash
# 启动 3 节点集群
bash start_local_cluster.sh

# 停止集群
bash stop_local_cluster.sh
```

### 4.2 手动启动节点

```bash
# 终端 1: 元数据服务
cd primihub
java -jar meta_service/meta-simple.jar \
  --server.port=8088 \
  --grpc.server.port=9099 \
  --db.path=/tmp/meta_node0 \
  --collaborate=""

# 终端 2: 节点 0
./bazel-bin/node --node_id=node0 --config=config/primihub_node0.yaml

# 终端 3: 节点 1
./bazel-bin/node --node_id=node1 --config=config/primihub_node1.yaml

# 终端 4: 节点 2
./bazel-bin/node --node_id=node2 --config=config/primihub_node2.yaml

# 终端 5: 提交任务
./bazel-bin/cli --task_config_file="example/psi_ecdh_task_conf.json"
```

### 4.3 节点配置文件

每个节点需要一个 YAML 配置文件，核心配置项：

```yaml
# config/primihub_node0.yaml
version: "1.0"
node: "node0"                              # 节点 ID
location: "192.168.1.100"                  # 节点 IP 或主机名
use_tls: false                             # 是否启用 TLS
grpc_port: 50050                           # gRPC 监听端口

# 代理配置（网络受限环境使用）
proxy_server:
  mode: "grpc"
  ip: "127.0.0.1"
  port: 50050
  use_tls: false

# 元数据服务连接
meta_service:
  mode: "grpc"
  ip: "192.168.1.100"                      # Meta 服务地址
  port: 9099
  use_tls: false

# 注册的数据集
datasets:
  - description: "psi_client_data"
    model: "csv"
    source: "data/client_e.csv"
```

---

## 5. Docker 部署

### 5.1 Docker-Compose 多节点部署

`docker-compose.yml` 定义了完整的 6 容器部署：

```yaml
version: '3'
services:
  node0:
    image: primihub/primihub-node:latest
    container_name: primihub-node0
    volumes:
      - ./data:/app/data
    entrypoint:
      - "GLOG_logtostderr=1 GLOG_v=2 ./primihub-node --node_id=node0 --config=/app/config/primihub_node0.yaml"
    depends_on:
      meta0:
        condition: service_healthy

  meta0:
    image: primihub/primihub-meta:latest
    container_name: primihub-meta0
    entrypoint:
      - "java -jar /applications/meta-simple.jar --server.port=8088 --grpc.server.port=9099 --db.path=/data/meta_service/node0"
    healthcheck:
      test: ["CMD-SHELL", "wget -O - -q http://localhost:8088/health"]
      interval: 8s
      retries: 3
```

部署命令：

```bash
# 标准部署
docker-compose up -d

# 扩展部署（自定义节点数）
docker-compose up -d --scale node=5 --scale meta=5

# 仅启动部分服务
docker-compose up -d node0 node1 meta0

# 使用特定网络
docker-compose --project-name primihub-prod up -d
```

### 5.2 Docker All-in-One 部署

`docker-all-in-one/` 提供了更完整的单机部署方案：

```bash
cd docker-all-in-one

# 修改配置（按需修改 .env）
vim .env

# 部署
bash deploy.sh

# 验证
docker-compose ps
```

### 5.3 构建自定义 Docker 镜像

```bash
# 完整构建
bash build_docker.sh FULL latest primihub/primihub-node

# 精简构建（测试用）
bash build_docker.sh MINI test primihub/primihub-node:test

# 本地构建
bash build_local.sh FULL latest primihub/primihub-node:local

# 多架构构建
docker buildx build --platform linux/amd64,linux/arm64 \
  -t primihub/primihub-node:latest --push .
```

### 5.4 镜像仓库

| 仓库地址 | 区域 | 说明 |
|---------|------|------|
| `docker.io/primihub/primihub-node` | 全球 | Docker Hub 官方 |
| `registry.cn-beijing.aliyuncs.com/primihub/primihub-node` | 中国 | 阿里云镜像 |

---

## 6. 配置指南

### 6.1 节点配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `node` | string | - | 节点唯一标识 |
| `location` | string | - | 节点位置/IP |
| `use_tls` | bool | false | 启用 TLS 加密 |
| `grpc_port` | int | 50050 | gRPC 服务端口 |
| `proxy_server.mode` | string | "grpc" | 代理模式 |
| `meta_service.ip` | string | - | 元数据服务地址 |
| `meta_service.port` | int | 9099 | 元数据服务端口 |

### 6.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GLOG_logtostderr` | 0 | 日志输出到 stderr |
| `GLOG_v` | 0 | 日志详细级别 (0-7) |
| `GLOG_log_dir` | ./log | 日志文件目录 |
| `REGISTRY` | docker.io | Docker 镜像仓库 |
| `TAG` | latest | Docker 镜像标签 |

### 6.3 TLS 配置

```yaml
# 在节点配置中启用 TLS
use_tls: true

certificate:
  root_ca: "data/cert/ca.crt"       # CA 证书
  key: "data/cert/node0.key"        # 节点私钥
  cert: "data/cert/node0.crt"       # 节点证书
```

生成自签名证书：

```bash
# 生成 CA 密钥和证书
openssl genrsa -out data/cert/ca.key 2048
openssl req -new -x509 -days 365 -key data/cert/ca.key -out data/cert/ca.crt

# 生成节点密钥和证书请求
openssl genrsa -out data/cert/node0.key 2048
openssl req -new -key data/cert/node0.key -out data/cert/node0.csr

# 签发节点证书
openssl x509 -req -days 365 -in data/cert/node0.csr \
  -CA data/cert/ca.crt -CAkey data/cert/ca.key -out data/cert/node0.crt
```

---

## 7. 使用指南

### 7.1 命令行 CLI

```bash
# PSI 隐私求交（ECDH 协议）
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"

# PSI 隐私求交（KKRT 协议）
./primihub-cli --task_config_file="example/psi_kkrt_task_conf.json"

# 隐私信息检索（PIR）
./primihub-cli --task_config_file="example/keyword_pir_task_conf.json"

# 安全多方计算 - 加法
./primihub-cli --task_config_file="example/mpc_add_task_conf.json"

# 安全多方计算 - 统计
./primihub-cli --task_config_file="example/mpc_statistics_task_conf.json"

# 联邦学习 - XGBoost
./primihub-cli --task_config_file="example/FL/xgboost/hetero_xgb.json"

# 联邦学习 - 逻辑回归
./primihub-cli --task_config_file="example/FL/logistic_regression/hfl_logistic_regression.json"

# 使用 TLS
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json" \
  --certificate="data/cert/ca.crt" \
  --key="data/cert/client.key" \
  --cert="data/cert/client.crt"
```

### 7.2 Python SDK

```python
from primihub import PrimiHubClient

# 创建客户端实例
client = PrimiHubClient()

# 初始化连接
client.init(config={
    "node": "127.0.0.1:50050",
    "cert": None   # TLS 证书路径（可选）
})

# 提交联邦学习任务
@ph.context.register
def train_model():
    from primihub.FL.xgboost import HeteroXGBoost
    model = HeteroXGBoost()
    model.train(
        num_tree=5,
        max_depth=5,
        learning_rate=0.1
    )

client.async_remote_execute(train_model)
client.start()
```

### 7.3 gRPC API

```python
import grpc
from primihub.protos import worker_pb2, worker_pb2_grpc

# 连接节点
channel = grpc.insecure_channel('localhost:50050')
stub = worker_pb2_grpc.VMNodeStub(channel)

# 构造 PSI 任务
task = worker_pb2.Task(
    type=worker_pb2.PSI_TASK,
    name="psi_ecdh_task",
    language=worker_pb2.PROTO,
    params=worker_pb2.Params(
        param_map={
            "psiTag": worker_pb2.ParamValue(var_type=worker_pb2.INT32, value_int32=0),
            "psiType": worker_pb2.ParamValue(var_type=worker_pb2.INT32, value_int32=0),
        }
    )
)

# 提交任务
request = worker_pb2.PushTaskRequest(task=task)
response = stub.SubmitTask(request)
print(f"Task submitted: {response.ret_code}")
```

### 7.4 数据集注册

```bash
# 通过 CLI 注册 CSV 数据集
./primihub-cli --register_dataset \
  --description="my_data" \
  --model=csv \
  --source="/path/to/data.csv"
```

---

## 8. 运维指南

### 8.1 日志管理

```bash
# 查看节点日志
docker logs -f primihub-node0

# 设置日志级别
docker exec primihub-node0 bash -c "GLOG_v=3 ./primihub-node ..."

# 日志文件位置（本地部署）
ls -la ./log/
```

日志清理脚本：

```bash
# scripts/cleanup_logs.sh
#!/bin/bash
LOG_DIR="./log"
find $LOG_DIR -name "*.log.*" -mtime +7 -delete
```

### 8.2 健康检查

```bash
# 检查元数据服务健康
curl http://localhost:8088/health

# 检查容器状态
docker ps --filter "name=primihub"

# 检查端口监听
netstat -tlnp | grep -E "50050|9099"
```

### 8.3 性能调优

```bash
# 1. 调整日志级别减少 IO
GLOG_v=0 ./primihub-node ...

# 2. 调整 gRPC 消息大小
export GRPC_MAX_RECV_MESSAGE_SIZE=104857600  # 100MB

# 3. 使用更快的编译优化
bazel build //:node --config=opt --copt="-O3" --copt="-march=native"

# 4. 增加并发 Worker 数
# 在节点配置中设置
# max_workers: 8
```

### 8.4 故障排查

| 症状 | 可能原因 | 排查步骤 |
|------|---------|---------|
| 容器启动失败 | 端口冲突 | `netstat -tlnp \| grep 50050` |
| 任务提交失败 | 节点未就绪 | `docker logs primihub-node0` |
| 节点无法通信 | 网络问题 | `docker exec primihub-node0 ping primihub-node1` |
| 元数据服务异常 | 数据库损坏 | `docker logs primihub-meta0` |
| PSI 结果为空 | 数据不匹配 | 检查输入数据格式 |
| 内存溢出 | 数据量过大 | 减少数据集或增加内存 |

### 8.5 备份与恢复

```bash
# 备份数据目录
tar -czf primihub_data_backup.tar.gz data/

# 备份配置文件
tar -czf primihub_config_backup.tar.gz config/

# 恢复
tar -xzf primihub_data_backup.tar.gz
docker-compose restart
```

---

## 9. Docker All-in-One 详细部署

### 9.1 目录结构

```
docker-all-in-one/
├── .env                    # 环境变量配置
├── docker-compose.yaml     # 容器编排
├── deploy.sh              # 部署脚本
├── README.md              # 说明文档
├── config/
│   ├── default0.conf       # 节点 0 配置
│   ├── default1.conf       # 节点 1 配置
│   ├── default2.conf       # 节点 2 配置
│   ├── primihub_node0.yaml # 节点 0 服务配置
│   ├── primihub_node1.yaml
│   ├── primihub_node2.yaml
│   ├── my.cnf             # MySQL 配置
│   └── redis.conf         # Redis 配置
├── data/
│   ├── env/               # 环境变量
│   │   ├── mysql.env
│   │   └── nacos-mysql.env
│   └── initsql/           # 初始化 SQL
│       ├── nacos_config.sql
│       ├── privacy1.sql
│       └── privacy2.sql
└── export_images.sh       # 镜像导出脚本
```

### 9.2 部署步骤

```bash
cd docker-all-in-one

# 1. 修改环境变量（按需）
vim .env

# 2. 执行部署
bash deploy.sh

# 3. 验证
docker-compose ps
docker-compose logs -f node0
```

---

## 10. 安全部署建议

### 10.1 生产环境安全配置

1. **启用 TLS 加密通信**
2. **使用强密码保护元数据服务**
3. **限制端口暴露范围**
4. **配置防火墙规则**
5. **定期更新镜像和依赖**
6. **启用日志审计**
7. **使用专用服务账户运行节点**

### 10.2 防火墙配置

```bash
# 仅开放必要端口
ufw allow 50050/tcp  # 节点 gRPC
ufw allow 9099/tcp   # 元数据服务
ufw deny 8088/tcp    # 健康检查限制内网
ufw enable
```

### 10.3 多节点网络拓扑

```
[数据中心网络]
     │
 ┌───┴───┐
 │  LB   │ (可选负载均衡)
 └───┬───┘
     │
 ┌───┼───────┐
 │   │       │
 ▼   ▼       ▼
N0──N1──N2  (节点集群，内网互联)
 │   │       │
 ▼   ▼       ▼
M0──M1──M2  (元数据集群)
```

---

## 11. 部署验证清单

| 检查项 | 命令 | 预期结果 |
|--------|------|---------|
| 容器运行 | `docker-compose ps` | 6 个容器均为 Up |
| 节点日志 | `docker logs primihub-node0` | 无 ERROR 日志 |
| 元数据健康 | `curl localhost:8088/health` | 返回 200 OK |
| PSI 功能 | `./primihub-cli --task_config_file=...` | 返回 SUCCESS |
| 结果文件 | `cat data/result/psi_result.csv` | 存在交集结果 |
