<p align="center">
  <img src="doc/header.jpeg" alt="Header">
  <br>

  <p align="center"><strong>由密码学专家团队打造的开源隐私计算平台</strong></p>

  <p align="center">
    <a href="https://github.com/primihub/primihub/releases"><img src="https://img.shields.io/github/v/release/primihub/primihub?style=flat-square" alt="GitHub Release"></a>
    <a href="https://github.com/primihub/primihub/actions/workflows/main.yml"><img src="https://img.shields.io/github/actions/workflow/status/primihub/primihub/main.yml?logo=github&style=flat-square" alt="Build Status"></a>
    <a href="https://hub.docker.com/r/primihub/primihub-node"><img src="https://img.shields.io/docker/pulls/primihub/primihub-node?style=flat-square" alt="Docker Pulls"></a>
  </p>
  
  <p align="center">
   中文 | <a href='README_EN.md'>English</a>
  </p>

</p>

隐私计算
-------

数据流动起来才可以创造更大的价值，随着数字经济持续高速增长，**数据的互联互通需求越来越旺盛**，大到政府机关的机密数据、公司核心商业数据、小到个人信息。近两年，我国也相继出台了 **《数据安全法》** 和 **《个人信息保护法》**。因此，**如何让数据安全地流通起来，是一个必须要解决的问题**。

隐私计算技术作为**连接数据流通和隐私保护法规的纽带**，实现了 **“数据可用不可见”**。即**在保护数据本身不对外泄露的前提下实现数据分析计算的技术集合**。隐私计算作为数据流通的**重要创新前沿技术**，已经广泛应用于金融、医疗、通信、政务等多个行业。

PrimiHub
-------

如果你对隐私计算感兴趣，想近距离体验下隐私计算的魅力，不妨试试 PrimiHub！一款**由密码学专家团队打造的开源隐私计算平台**，它安全可靠、开箱即用、自主研发、功能丰富。

特性
---

* **开源**：完全开源、免费，Apache 2.0 许可证
* **安装简单**：支持 Docker 一键部署，支持 x86_64 和 ARM64 架构
* **开箱即用**：拥有 [Web界面](https://github.com/primihub/primihub-platform)、[命令行](https://docs.primihub.com/docs/category/%E5%88%9B%E5%BB%BA%E4%BB%BB%E5%8A%A1) 和 [Python SDK](https://docs.primihub.com/docs/category/python-sdk-client) 多种使用方式
* **功能丰富**：支持隐匿查询、隐私求交、联合统计、数据资源管理等功能
* **灵活配置**：支持自定义扩展语法、语义、安全协议等
* **自主研发**：基于安全多方计算、联邦学习、同态加密、可信计算等隐私计算技术

核心功能
-------

### 🔐 隐私求交 (PSI)

隐私求交技术允许多方在不泄露各自数据的前提下，计算数据集的交集。

**支持的算法**：
- **ECDH-PSI**：基于椭圆曲线 Diffie-Hellman 密钥交换
- **KKRT-PSI**：高效的 KKRT 协议实现
- **TEE-PSI**：基于可信执行环境的隐私求交

**应用场景**：
- 金融风控：多机构联合黑名单查询
- 精准营销：广告主与媒体的用户匹配
- 医疗健康：跨医院患者数据匹配

### 🔍 隐匿查询 (PIR)

隐匿查询技术允许用户在不泄露查询内容的前提下，从数据库中检索信息。

**支持的类型**：
- **ID-PIR**：基于唯一标识符的查询
- **Keyword-PIR**：基于关键字的查询

**应用场景**：
- 信用查询：保护查询者隐私
- 数据检索：隐私保护的数据库访问

### 🤖 联邦学习 (FL)

联邦学习在保护数据隐私的前提下，实现多方联合训练机器学习模型。

**学习模式**：
- **横向联邦学习 (HFL)**：适用于特征相同、样本不同的场景
- **纵向联邦学习 (VFL)**：适用于样本相同、特征不同的场景

**支持的算法**：
| 算法类型 | HFL | VFL |
|---------|-----|-----|
| 线性回归 | ✅ | ✅ |
| 逻辑回归 | ✅ | ✅ |
| 神经网络 | ✅ | ❌ |
| CNN | ✅ | ❌ |
| XGBoost | ✅ | ✅ |
| ChatGLM | ✅ | ❌ |

**应用场景**：
- 金融风控：多机构联合建模
- 医疗诊断：跨医院疾病预测
- 智能推荐：多平台协同推荐

### 🔢 安全多方计算 (MPC)

安全多方计算允许多方在不泄露各自输入的前提下，共同计算某个函数。

**支持的功能**：
- **联合统计**：多方数据的统计分析
- **隐私求和/求平均**：保护隐私的数值计算
- **表达式计算**：自定义计算逻辑

**应用场景**：
- 薪资统计：隐私保护的薪资水平分析
- 数据分析：多方联合数据分析

### 🔐 同态加密

同态加密允许直接对加密数据进行计算，无需解密。

**支持的算法**：
- **Paillier 加密**：支持加法同态
- **CKKS 加密**：支持浮点数近似计算

**应用场景**：
- 加密计算：云端加密数据处理
- 隐私保护：敏感数据加密存储和计算

### 🛡️ 可信执行环境 (TEE)

基于硬件的可信执行环境，提供更高的安全保障。

**支持的技术**：
- **Intel SGX**：硬件级别的安全保护

**应用场景**：
- 高安全场景：核心数据保护
- 混合方案：TEE + 密码学协议

### 📊 数据管理

灵活的数据源支持和管理能力。

**支持的数据源**：
- CSV 文件
- SQLite 数据库
- MySQL 数据库
- HDFS 分布式文件系统
- 图像文件

**数据预处理**：
- 数据对齐
- 特征工程
- IV/WOE 分析
- 数据标准化

快速开始
-------

推荐使用 Docker 部署 PrimiHub，开启你的隐私计算之旅。

```
# 第一步：下载
git clone https://github.com/primihub/primihub.git
# 第二步：启动容器
cd primihub && docker-compose up -d
# 第三步：进入容器
docker exec -it primihub-node0 bash
# 第四步：执行隐私求交计算
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
I20230616 13:40:10.683375    28 cli.cc:524] all node has finished
I20230616 13:40:10.683745    28 cli.cc:598] SubmitTask time cost(ms): 1419
# 查看结果
cat data/result/psi_result.csv
"intersection_row"
X3
...
```

<p align="center"><img src="doc/kt.gif" width=700 alt="PSI"></p>

<p align="center"><em>隐私求交例子 <a href="https://docs.primihub.com/docs/quick-start-platform/">在线尝试</a>・<a href="https://docs.primihub.com/docs/advance-usage/create-tasks/psi-task/">命令行</a></em></p>

除此之外，PrimiHub 还提供了多种适合**不同人群**的使用方式：

* [在线体验](https://docs.primihub.com/docs/quick-start-platform/)
* [Docker](https://docs.primihub.com/docs/advance-usage/start/quick-start)
* [可执行文件](https://docs.primihub.com/docs/advance-usage/start/start-nodes)
* [自行编译](https://docs.primihub.com/docs/advance-usage/start/build)

技术架构
-------

### 核心组件

```
┌─────────────────────────────────────────────────────────┐
│                     应用层 (Application Layer)            │
│  Web UI  │  CLI 命令行  │  Python SDK  │  gRPC API      │
├─────────────────────────────────────────────────────────┤
│                     任务调度层 (Task Layer)               │
│  任务解析  │  语义分析  │  任务编排  │  资源调度         │
├─────────────────────────────────────────────────────────┤
│                    算法层 (Algorithm Layer)               │
│  PSI  │  PIR  │  联邦学习  │  MPC  │  同态加密  │ TEE   │
├─────────────────────────────────────────────────────────┤
│                    通信层 (Communication Layer)           │
│  gRPC 服务  │  P2P 网络  │  消息路由  │  数据传输       │
├─────────────────────────────────────────────────────────┤
│                    数据层 (Data Layer)                    │
│  数据抽象  │  多源接入  │  元数据管理  │  数据驱动       │
└─────────────────────────────────────────────────────────┘
```

### 技术栈

| 层级 | 技术 |
|------|------|
| **核心引擎** | C++17, Bazel |
| **算法库** | 自研密码学库, Intel SGX |
| **Python 框架** | Python 3.8+, TensorFlow, PyTorch, XGBoost |
| **通信协议** | gRPC, Protocol Buffers |
| **数据存储** | CSV, SQLite, MySQL, HDFS |
| **容器化** | Docker, Docker Compose |

开发与构建
---------

### 构建镜像

PrimiHub 提供了三种构建脚本，详细说明请参考 [BUILD_SCRIPTS.md](BUILD_SCRIPTS.md)：

```bash
# 方式 1: 基础本地构建
bash build_local.sh [MODE] [TAG] [IMAGE_NAME]

# 方式 2: 完整构建（推荐）
bash build_docker.sh FULL latest primihub/primihub-node

# 方式 3: CI/CD 构建（Jenkins）
bash jenkins_build.sh ALL 192.168.99.10
```

**编译模式**：
- `FULL/ALL`: 完整版本，适用于生产环境
- `MINI`: 精简版本，适用于测试环境

### 从源码编译

```bash
# 1. 安装依赖
bash pre_build.sh

# 2. 编译项目
make mysql=y

# 3. 运行节点
./primihub-node --node_id=node0 --config=config/primihub_node0.yaml
```

### 开发指南

**项目结构**：
```
primihub/
├── src/primihub/           # C++ 核心实现
│   ├── algorithm/          # 算法实现
│   ├── kernel/             # PSI/PIR 内核
│   ├── node/               # 节点服务
│   └── task/               # 任务框架
├── python/primihub/        # Python SDK
│   ├── FL/                 # 联邦学习
│   ├── MPC/                # 安全多方计算
│   └── client/             # 客户端
├── example/                # 示例配置
├── config/                 # 节点配置
└── test/                   # 测试用例
```

**代码规范**：
- C++: Google C++ Style Guide
- Python: PEP 8
- 提交前运行测试: `bazel test //...`

使用示例
-------

### 命令行 CLI

```bash
# 隐私求交
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"

# 联邦学习
./primihub-cli --task_config_file="example/FL/xgboost/hetero_xgb.json"
```

### Python SDK

```python
from primihub import PrimiHubClient

# 创建客户端
client = PrimiHubClient()

# 提交任务
task_config = {
    "party_info": {
        "task_manager": "127.0.0.1:50050"
    },
    "component_params": {
        "roles": {"guest": ["Bob"], "host": ["Charlie"]},
        "common_params": {"model": "HFL_logistic_regression"}
    }
}

client.submit_task(task_config)
```

### gRPC API

```python
import grpc
from primihub.protos import worker_pb2, worker_pb2_grpc

# 连接节点
channel = grpc.insecure_channel('localhost:50050')
stub = worker_pb2_grpc.VMNodeStub(channel)

# 提交任务
request = worker_pb2.PushTaskRequest(...)
response = stub.SubmitTask(request)
```

问题 / 帮助 / Bug
------------

如果您在使用过程中遇到任何问题，需要我们的帮助可以 [点击](https://github.com/primihub/primihub/issues/new/choose) 反馈问题。

欢迎添加我们的微信助手，加入「PrimiHub 开源社区」微信群。“零距离”接触**项目核心开发、密码学专家、隐私计算行业大咖**，获得更及时的回复和隐私计算的第一手资讯。

<p align="center">
  <img src="doc/wechat.jpeg" alt="Header">
</p>

许可证
-----

此代码在 Apache 2.0 下发布，参见 [LICENSE](https://github.com/primihub/primihub/blob/develop/LICENSE) 文件。
