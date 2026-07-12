# PrimiHub 隐私计算平台 — 项目分析

## 一、项目概述

**PrimiHub** 是一个开源的隐私计算平台，由密码学专家团队开发。核心理念是**数据可用不可见**——在不暴露原始数据的前提下，实现多方数据的安全协作计算。基于 Apache 2.0 协议开源。

- **组织**: [PrimiHub](https://github.com/primihub/primihub)
- **网站**: https://www.primihub.com
- **邮箱**: openmpc@primihub.com
- **编程语言**: C++17（核心引擎）、Python（SDK 与算法）、Java（元数据服务）

---

## 二、系统架构

### 2.1 五层架构

```
┌──────────────────────────────────────────────────────────┐
│                    应用层 (Application)                    │
│     Web UI  │  CLI  │  Python SDK  │  gRPC API            │
├──────────────────────────────────────────────────────────┤
│                    任务调度层 (Scheduling)                 │
│     任务解析  │  语义分析  │  任务编排  │  资源调度          │
├──────────────────────────────────────────────────────────┤
│                    算法层 (Algorithm)                      │
│   PSI  │  PIR  │  FL  │  MPC  │  HE  │  TEE              │
├──────────────────────────────────────────────────────────┤
│                    通信层 (Communication)                  │
│   gRPC  │  P2P 网络  │  消息路由  │  数据传输              │
├──────────────────────────────────────────────────────────┤
│                    数据层 (Data)                           │
│   数据抽象  │  多源接入  │  元数据管理  │  数据预处理         │
└──────────────────────────────────────────────────────────┘
```

### 2.2 核心设计理念

- **节点式分布式系统**: 多个 `primihub-node` 实例组成 P2P 网络
- **元数据服务**: Java 服务负责数据集注册与发现
- **任务驱动执行**: 通过 JSON 配置文件定义任务，调度器分发执行
- **可插拔算法**: 支持 C++（原生）和 Python（SDK）两种算法扩展方式
- **通信协议**: gRPC + Apache Arrow Flight

---

## 三、项目目录结构

```
primihub/
├── src/primihub/           # C++ 核心引擎
│   ├── algorithm/          # 隐私保护算法实现
│   ├── cli/                # 命令行客户端
│   ├── common/             # 公共类型、配置、参与方管理
│   ├── data_store/         # 多源数据访问层
│   ├── executor/           # MPC 统计/表达式执行器
│   ├── kernel/             # PSI/PIR 内核实现
│   ├── node/               # 节点服务入口与实现
│   ├── operator/           # ABY3 MPC 算子
│   ├── p2p/                # P2P 网络通信
│   ├── protos/             # Protocol Buffers 定义
│   ├── pybind_warpper/     # C++/Python 绑定 (pybind11)
│   ├── service/            # 数据集与通知服务
│   ├── sgx/                # Intel SGX TEE 组件
│   ├── task/               # 任务框架与语义层
│   ├── task_engine/        # 任务执行引擎
│   └── util/               # 工具函数
│
├── python/primihub/        # Python SDK
│   ├── client/             # Python 客户端
│   ├── FL/                 # 联邦学习
│   │   ├── linear_regression/    # 线性回归
│   │   ├── logistic_regression/  # 逻辑回归
│   │   ├── neural_network/       # 神经网络
│   │   ├── xgboost/              # XGBoost
│   │   ├── chatglm/              # ChatGLM 大模型
│   │   ├── crypto/               # 密码学原语
│   │   ├── preprocessing/        # 数据预处理
│   │   ├── psi/                  # PSI 集成
│   │   ├── sketch/               # 数据素描 (Theta Sketch)
│   │   └── stats/                # 统计检验
│   ├── MPC/                # MPC Python 封装
│   ├── TEE/                # TEE 支持
│   ├── dataset/            # 数据集客户端
│   ├── engine/             # Python 任务引擎
│   ├── sql_security/       # SQL 安全
│   │   ├── validators/     # 校验器 (聚合/过滤/分组/连接/排序/子查询/窗口)
│   │   └── functions/      # 函数安全 (日期/数值/字符串/时间戳)
│   └── utils/              # 工具
│
├── config/                 # 节点配置文件
├── example/                # 任务配置示例 (JSON)
├── data/                   # 测试数据
├── doc/                    # 文档图片
├── docs/                   # 文档
├── test/                   # C++ 测试
├── e2etest/                # 端到端集成测试
├── third_party/            # 第三方 C++ 库
├── docker-all-in-one/      # Docker 全合一部署
├── docker-one-in-one/      # Docker 单节点部署
├── scripts/                # 运维脚本
└── *.sh                    # 构建/部署脚本
```

---

## 四、核心功能

### 4.1 隐私集合求交 (PSI)

| 协议 | 描述 |
|------|------|
| ECDH-PSI | 基于椭圆曲线 Diffie-Hellman 密钥交换 |
| KKRT-PSI | 高效的 KKRT 协议实现 |
| TEE-PSI | 基于可信执行环境的 PSI |

支持交集/差集模式、列选择、结果同步。

### 4.2 隐私信息检索 (PIR)

| 类型 | 描述 |
|------|------|
| ID-PIR | 基于标识符的隐私检索 |
| Keyword-PIR | 基于关键词的隐私检索 |
| APSI | Microsoft APSI 集成 |
| PIR-ACC | 加速 PIR（自定义 SIGMA 协议，GPU 加速） |

### 4.3 安全多方计算 (MPC)

- **协议**: ABY3（三方安全计算）
- **基础算子**: ADD、SUB、MUL、DIV、CMP
- **统计运算**: SUM、AVG、MAX、MIN
- **统计检验**: T_TEST、F_TEST、CHI_SQUARE_TEST
- **高级分析**: 回归分析、相关性分析

### 4.4 联邦学习 (FL)

| 算法 | 横向联邦 | 纵向联邦 |
|------|---------|---------|
| 线性回归 | ✓ | ✓ |
| 逻辑回归 | ✓ | ✓ |
| 神经网络 | ✓ | ✗ |
| CNN | ✓ | ✗ |
| XGBoost | ✓ | ✓ |
| ChatGLM 微调 | ✓ | ✗ |

- **差分隐私**: 集成 Opacus 库
- **安全聚合**: 加密聚合模型更新
- **多方训练**: 支持 3 方以上参与方

### 4.5 同态加密 (HE)

| 方案 | 类型 | 用途 |
|------|------|------|
| Paillier | 加法同态 | 加密计算 |
| CKKS | 近似浮点同态 | 机器学习 |

### 4.6 可信执行环境 (TEE)

- **Intel SGX**: 硬件级内存隔离
- **远程证明 (RA)**: TEE 完整性验证
- **RaTLS**: 带远程证明的 TLS

### 4.7 数据管理

- **多源接入**: CSV、SQLite、MySQL、HDFS、Parquet、图片
- **数据预处理**: 数据对齐、特征工程、IV/WOE 分析、标准化
- **数据集注册**: 通过元数据服务管理

### 4.8 SQL 安全

SQL 语句的安全校验，覆盖以下维度：
- **聚合函数校验**: 防止敏感聚合泄露
- **过滤条件校验**: 确保过滤条件合理
- **分组校验**: 控制分组粒度
- **连接校验**: 限制表连接范围
- **排序校验**: 防止排序列泄露
- **子查询校验**: 子查询安全管控
- **窗口函数校验**: 窗口函数安全使用

---

## 五、构建系统

### 5.1 主构建系统: Bazel

- **版本要求**: Bazel 5.0.0+
- **C++ 标准**: C++17
- **支持平台**:
  - Linux x86_64（SSE/AVX 优化）
  - Linux aarch64（ARM64）
  - macOS Intel / Apple Silicon
  - Windows（MSVC，部分支持）

### 5.2 构建目标

| 目标 | 产物 | 描述 |
|------|------|------|
| `//:node` | `primihub-node` | 主节点二进制 |
| `//:cli` | `primihub-cli` | CLI 客户端 |
| `//:task_main` | `task_main` | 任务执行子进程 |
| `linkcontext` | `linkcontext.so` | pybind11 桥梁 |
| `opt_paillier_c2py.so` | Python Paillier 绑定 |

### 5.3 快捷构建

```bash
make release              # 发布构建
make mysql=y              # 启用 MySQL 驱动
make debug=y              # ASAN 调试模式
make tee=y                # 启用 SGX TEE
make jobs=4               # 并行编译
```

### 5.4 依赖管理

依赖通过 `bazel/repository_deps.bzl` 统一管理，支持 GitHub 和国内镜像（Gitee/阿里云 OSS）两种来源，通过 `WORKSPACE` / `WORKSPACE_GITHUB` / `WORKSPACE_CN` 切换。

---

## 六、部署方式

### 6.1 Docker Compose（推荐快速开始）

```bash
docker-compose up -d
```
启动 6 个容器：3 个计算节点 + 3 个元数据服务。

### 6.2 本地部署

```bash
bash pre_build.sh
make mysql=y

# 分别启动 3 个节点
./primihub-node --node_id=node0 --config=config/primihub_node0.yaml
./primihub-node --node_id=node1 --config=config/primihub_node1.yaml
./primihub-node --node_id=node2 --config=config/primihub_node2.yaml

# 提交任务
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
```

### 6.3 Docker 镜像

- **Docker Hub**: `primihub/primihub-node`
- **阿里云镜像**: `registry.cn-beijing.aliyuncs.com/primihub/primihub-node`

---

## 七、使用示例

### PSI（隐私集合求交）

```bash
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
# 结果输出: data/result/psi_result.csv
```

### MPC 加法

```bash
./primihub-cli --task_config_file="example/mpc_add_task_conf.json"
```

### 联邦学习

```bash
# XGBoost
./primihub-cli --task_config_file="example/FL/xgboost/hetero_xgb.json"

# 逻辑回归
./primihub-cli --task_config_file="example/FL/logistic_regression/hfl_logistic_regression.json"
```

### PIR（隐私信息检索）

```bash
./primihub-cli --task_config_file="example/keyword_pir_task_conf.json"
```

### Python SDK

```python
from primihub import PrimiHubClient

client = PrimiHubClient()
client.init(config={"node": "127.0.0.1:50050"})
client.async_remote_execute(...)
```

---

## 八、通信协议

### 8.1 gRPC 服务

**VMNode 服务** (`worker.proto`):
- `SubmitTask` / `ExecuteTask` — 提交/执行任务
- `KillTask` / `StopTask` — 任务生命周期管理
- `FetchTaskStatus` / `UpdateTaskStatus` — 状态查询
- `Send` / `Recv` / `SendRecv` — 多方数据交换
- `ForwardSend` / `ForwardRecv` — 代理转发

**DataSetService** (`service.proto`):
- `NewDataset` — 注册/注销数据集
- `GetDataset` — 查询数据集元数据
- `DownloadData` / `UploadData` — 数据传输

### 8.2 安全通信

- 可选 TLS 加密
- 证书认证（CA + 服务端 + 客户端）
- mTLS 双向认证

---

## 九、测试体系

### C++ 测试

基于 Google Test，通过 Bazel 运行：

```bash
bazel test //test/primihub/algorithm:logistic_test
bazel test //test/primihub/common:common_test
bazel test //test/primihub/executor:executor_test
```

### Python 端到端测试

```bash
cd e2etest/local
python start_node.py           # 启动测试节点
python test_fl.py              # FL 测试
python test_mpc.py             # MPC 测试
python test_psi.py             # PSI 测试
python test_pir.py             # PIR 测试
```

---

## 十、CI/CD

### GitHub Actions

- **main.yml**: PR 触发，在自托管 Ubuntu 上构建并运行测试
- **release.yaml**: Tag 触发，跨 4 平台构建（amd64/arm64 的 Linux/macOS）
  - 生成平台压缩包并上传到 GitHub Release
  - 构建并发布多架构 Docker 镜像

---

## 十一、安全特性

1. **密码学协议**: ECDH、ABY3、Paillier、CKKS 多层加密
2. **传输安全**: TLS 加密 + 证书认证
3. **差分隐私**: Opacus 集成 + 隐私预算追踪
4. **TEE**: Intel SGX 硬件隔离 + 远程证明
5. **数据安全**: 原始数据不离开节点，全密态计算
6. **SQL 安全**: 全方位的 SQL 语句安全校验

---

## 十二、技术栈汇总

| 层级 | 技术 |
|------|------|
| 核心语言 | C++17 |
| SDK 语言 | Python 3.8+ |
| 元数据服务 | Java |
| 构建系统 | Bazel 5.0+ |
| 通信 | gRPC + Apache Arrow Flight |
| 序列化 | Protocol Buffers |
| 日志 | glog (C++) / loguru (Python) |
| 数据库 | MySQL、SQLite、LevelDB |
| 加密库 | OpenSSL、RELIC、SEAL、GMP |
| 深度学习 | PyTorch、TensorFlow、XGBoost |
| 容器化 | Docker、Docker Compose |
