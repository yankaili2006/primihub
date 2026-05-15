# PrimiHub 架构设计文档

## 1. 系统架构概述

PrimiHub 采用**五层分层架构**和**节点式分布式部署**相结合的设计。系统由多个 `primihub-node` 计算节点组成 P2P 网络，通过元数据服务实现数据集的注册与发现，通过任务调度引擎统筹多节点协同计算。

### 1.1 架构总览

```
┌──────────────────────────────────────────────────────────────────┐
│                        应用层 (Application Layer)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  Web UI  │  │   CLI    │  │Python SDK│  │   gRPC API       │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                    任务调度层 (Task Scheduling Layer)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 任务解析  │  │ 语义分析  │  │ 任务编排  │  │   资源调度       │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                      算法层 (Algorithm Layer)                      │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────────┐ │
│  │ PSI  │ │ PIR  │ │  FL  │ │ MPC  │ │  HE  │ │     TEE      │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                    通信层 (Communication Layer)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │  gRPC    │  │ P2P 网络  │  │ 消息路由  │  │   数据传输      │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│                      数据层 (Data Layer)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 数据抽象  │  │ 多源接入  │  │ 元数据管理 │  │   数据预处理    │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 分布式节点拓扑

```
                    ┌─────────────────┐
                    │   CLIENT        │
                    │  (primihub-cli) │
                    └────────┬────────┘
                             │ gRPC
                             ▼
    ┌─────────────────────────────────────────────────┐
    │                   Meta Service                   │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
    │  │  meta0   │  │  meta1   │  │  meta2   │      │
    │  └──────────┘  └──────────┘  └──────────┘      │
    └─────────────────────────────────────────────────┘
              ▲              ▲              ▲
              │ gRPC         │ gRPC         │ gRPC
    ┌─────────┴──┐    ┌──────┴──────┐    ┌──┴─────────┐
    │  node0     │◄──►│   node1     │◄──►│   node2    │
    │  port:50050│    │  port:50051 │    │ port:50052 │
    └────────────┘    └─────────────┘    └────────────┘
         │                  │                  │
    ┌────┴────┐       ┌────┴────┐        ┌────┴────┐
    │ 数据源   │       │ 数据源   │        │ 数据源   │
    │ CSV/DB  │       │ CSV/DB  │        │ CSV/DB  │
    └─────────┘       └─────────┘        └─────────┘
```

---

## 2. 模块划分

### 2.1 计算节点 (primihub-node)

计算节点是系统的核心组件，每个节点包含以下子模块：

| 子模块 | 路径 | 职责 |
|--------|------|------|
| 节点服务 | `src/primihub/node/` | gRPC 服务入口、节点生命周期管理、Worker 管理 |
| 任务引擎 | `src/primihub/task_engine/` | 任务调度、执行、状态追踪 |
| 语义层 | `src/primihub/task/semantic/` | 任务解析、语义验证、参数映射 |
| 算法实现 | `src/primihub/algorithm/` | 各类隐私计算算法实现 |
| 密码学内核 | `src/primihub/kernel/` | PSI/PIR 密码协议内核 |
| 数据存储 | `src/primihub/data_store/` | 多源数据接入抽象层 |
| 通信层 | `src/primihub/p2p/` | 节点间 gRPC 通信 |
| 服务层 | `src/primihub/service/` | 数据集管理、通知服务 |
| SGX TEE | `src/primihub/sgx/` | Intel SGX 可信执行环境 |
| 工具库 | `src/primihub/util/` | 日志、网络、哈希等通用工具 |

### 2.2 命令行客户端 (primihub-cli)

| 子模块 | 路径 | 职责 |
|--------|------|------|
| CLI 入口 | `src/primihub/cli/main.cc` | 命令行参数解析 |
| 任务配置解析 | `src/primihub/cli/task_config_parser.cc` | 解析 JSON 任务配置 |
| 数据集配置解析 | `src/primihub/cli/dataset_config_parser.cc` | 解析数据集注册配置 |
| 核心逻辑 | `src/primihub/cli/cli.cc` | 任务提交、结果获取 |

### 2.3 Python SDK

| 子模块 | 路径 | 职责 |
|--------|------|------|
| 客户端 | `python/primihub/client/` | Python 客户端 SDK |
| 联邦学习 | `python/primihub/FL/` | FL 算法实现 |
| MPC 封装 | `python/primihub/MPC/` | MPC Python 接口 |
| TEE 支持 | `python/primihub/TEE/` | TEE 相关操作 |
| 数据集 | `python/primihub/dataset/` | 数据集客户端 |
| 任务引擎 | `python/primihub/engine/` | Python 端任务执行 |
| SQL 安全 | `python/primihub/sql_security/` | SQL 安全校验 |

### 2.4 元数据服务 (Meta Service)

元数据服务是一个轻量级 Java 服务，负责：
- 数据集注册与注销
- 数据集元数据查询
- 多节点间数据协作协调
- 节点健康检查

---

## 3. 通信协议

### 3.1 gRPC 服务定义

**VMNode 服务** — 节点间核心 RPC 服务：

```protobuf
service VMNode {
  // 任务操作
  rpc SubmitTask(PushTaskRequest) returns (PushTaskReply);
  rpc KillTask(KillTaskRequest) returns (KillTaskResponse);
  rpc ExecuteTask(PushTaskRequest) returns (PushTaskReply);
  rpc StopTask(TaskContext) returns (Empty);

  // 任务状态
  rpc FetchTaskStatus(TaskContext) returns (TaskStatusReply);
  rpc UpdateTaskStatus(TaskStatus) returns (Empty);

  // 数据交换
  rpc Send(stream TaskRequest) returns (TaskResponse);
  rpc Recv(TaskRequest) returns (stream TaskResponse);
  rpc SendRecv(stream TaskRequest) returns (stream TaskResponse);
  rpc ForwardSend(stream ForwardTaskRequest) returns (TaskResponse);
  rpc ForwardRecv(TaskRequest) returns (stream TaskRequest);
  rpc CompleteStatus(CompleteStatusRequest) returns (Empty);
}
```

**DataSetService** — 数据集管理服务：

```protobuf
service DataSetService {
  rpc NewDataset(NewDatasetRequest) returns (NewDatasetResponse);
  rpc GetDataset(GetDatasetRequest) returns (GetDatasetResponse);
  rpc QueryResult(QueryResultRequest) returns (QueryResultResponse);
  rpc DownloadData(DownloadRequest) returns (stream DownloadRespone);
  rpc UploadData(stream UploadFileRequest) returns (UploadFileResponse);
}
```

### 3.2 数据传输流程

```
CLIENT                  NODE0                    NODE1
   │                      │                        │
   │──SubmitTask()───────►│                        │
   │                      │──ForwardSend()────────►│
   │                      │                        │
   │                      │◄───────Send()─────────│
   │                      │                        │
   │◄──PushTaskReply()───│                        │
   │                      │                        │
   │──Send()─────────────►│                        │
   │                      │──ForwardSend()────────►│
   │                      │                        │
   │◄─────Recv()─────────│                        │
   │                      │◄───────Recv()─────────│
```

### 3.3 数据交换模式

系统支持三种数据交换模式：

1. **直连模式 (Direct)**: 节点之间直接建立 gRPC 流进行数据交换
2. **代理转发模式 (Proxy)**: 通过中间节点转发数据，适用于网络受限环境
3. **流式传输 (Streaming)**: 支持大数据的流式分块传输

---

## 4. 任务执行流程

### 4.1 任务提交流程

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  CLIENT  │    │  NODE0   │    │  NODE1   │    │  NODE2   │
├─────────┤    ├──────────┤    ├──────────┤    ├──────────┤
│SubmitTask│───►│          │    │          │    │          │
│         │    │解析任务配置│    │          │    │          │
│         │    │解析参与方 │    │          │    │          │
│         │    │ForwardSend|──►│ForwardSend|──►│          │
│         │    │          │    │          │    │          │
│         │    │←─────────│Send│◄─────────│    │          │
│         │    │ 确认参与   │    │          │    │          │
│PushTask │◄───│          │    │          │    │          │
│  Reply  │    │          │    │          │    │          │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
```

### 4.2 任务执行流程（以 PSI 为例）

```
NODE0 (Client Party)                     NODE1 (Server Party)
         │                                      │
         │  ┌─ 任务解析 ─┐                      │
         │  │ 读取配置   │                      │
         │  │ 解析参数   │                      │
         │  └───────────┘                      │
         │                                      │
         │  ┌─ 语义分析 ─┐                      │
         │  │ 识别为PSI  │                      │
         │  │ 创建调度器 │                      │
         │  └───────────┘                      │
         │                                      │
         │  ┌─ 资源调度 ─┐                      │
         │  │ 分配worker │                      │
         │  │ 准备数据集  │                      │
         │  └───────────┘                      │
         │                                      │
         │  ──── 执行 PSI 协议 ────────────────►│
         │       ECDH 密钥交换                   │
         │◄──── 加密数据交换 ──────────────────│
         │      交集计算                         │
         │      结果保存                         │
         │                                      │
         │  ┌─ 返回结果 ─┐                      │
         │  │ 保存CSV   │                      │
         │  │ 通知完成   │                      │
         │  └───────────┘                      │
```

### 4.3 任务调度器实现

每个任务类型有对应的语义调度器：

| 任务类型 | 调度器类 | 路径 |
|---------|---------|------|
| PSI | `PSITask` | `src/primihub/task/semantic/psi_task.h` |
| PIR | `PIRTask` | `src/primihub/task/semantic/pir_task.h` |
| MPC | `MPCTask` | `src/primihub/task/semantic/mpc_task.h` |
| FL | `FLTask` | `src/primihub/task/semantic/fl_task.h` |
| TEE | `TEETask` | `src/primihub/task/semantic/tee_task.h` |

---

## 5. 技术选型

### 5.1 核心技术栈

| 层级 | 技术选型 | 选型理由 |
|------|---------|---------|
| 核心引擎语言 | C++17 | 高性能、底层控制、丰富的密码学库支持 |
| SDK 语言 | Python 3.8+ | 数据科学生态丰富、开发效率高 |
| 元数据服务 | Java (Spring Boot) | 成熟的企业级服务框架 |
| 构建工具 | Bazel 5.0+ | 多语言支持、精确的依赖管理、缓存加速 |
| RPC 通信 | gRPC | 高性能、双向流、Protocol Buffers |
| 数据格式 | Apache Arrow | 列式内存格式、零拷贝数据传输 |
| 序列化 | Protocol Buffers | 高效的二进制序列化、跨语言 |
| 密码学库 | RELIC、SEAL、OpenSSL | 成熟、经过安全审计 |
| 深度学习 | PyTorch、XGBoost | 主流 ML/DL 框架 |
| 容器化 | Docker + Compose | 标准化部署、环境隔离 |

### 5.2 主要第三方依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| gRPC | 1.42.x | 节点间通信 |
| Protocol Buffers | 3.20.0 | 消息序列化 |
| Apache Arrow | 4.0.0 | 列式数据格式 |
| Eigen | 3.4 | 线性代数 |
| ABY3 | - | 三方安全计算 |
| Microsoft SEAL | - | 同态加密 (CKKS) |
| **HEhub** (Git 子模块) | - | 自研 HE 库: BGV + CKKS + TFHE |
| pybind11 | 2.9.2 | C++/Python 绑定 |
| OpenSSL | 1.1.1 | 加密/TLS |
| Intel SGX SDK | - | TEE 支持 |
| Redis | - | 状态缓存 |

---

## 6. 部署架构

### 6.1 最小部署（3 节点）

```
┌─────────────────────────────────────────────┐
│                  Docker Host                 │
│                                              │
│  ┌──────┐  ┌──────┐  ┌──────┐              │
│  │node0 │  │node1 │  │node2 │              │
│  │:50050│  │:50051│  │:50052│              │
│  └──┬───┘  └──┬───┘  └──┬───┘              │
│     │         │         │                   │
│  ┌──┴────┐ ┌──┴────┐ ┌──┴────┐             │
│  │meta0  │ │meta1  │ │meta2  │             │
│  │:9099  │ │:9099  │ │:9099  │             │
│  └───────┘ └───────┘ └───────┘             │
│                                              │
│  ┌──────────────────────────────────┐       │
│  │         primihub_net (bridge)    │       │
│  └──────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

### 6.2 生产部署建议

- 每个节点部署在不同物理机或 VM 上
- 元数据服务与计算节点分离
- 使用反向代理暴露 gRPC 端口
- 启用 TLS 加密通信
- 配置日志轮转和集中式日志收集
- 部署监控告警系统

---

## 7. 安全架构

### 7.1 多层安全防护

```
┌─────────────────────────────────────────┐
│          应用层安全                        │
│  - 数据集可见性控制 (PUBLIC/PRIVATE)     │
│  - SQL 安全校验                          │
├─────────────────────────────────────────┤
│          通信层安全                        │
│  - TLS 1.3 加密                          │
│  - mTLS 双向证书认证                     │
│  - RaTLS (TEE 远程证明)                  │
├─────────────────────────────────────────┤
│          计算层安全                        │
│  - MPC 协议保证计算安全                  │
│  - FL 安全聚合                           │
│  - 差分隐私                              │
├─────────────────────────────────────────┤
│          硬件层安全                        │
│  - Intel SGX 可信执行环境                │
│  - 远程证明 (Remote Attestation)         │
└─────────────────────────────────────────┘
```

### 7.2 数据生命周期安全

| 阶段 | 安全措施 |
|------|---------|
| 数据接入 | 多源数据加密存储 |
| 数据传输 | gRPC + TLS 加密 |
| 计算过程 | MPC/HE 协议下的密态计算 (含 HEhub BGV/CKKS/TFHE 原生支持) |
| 结果输出 | 可配置的差隐私噪声添加 |
| 数据清理 | 计算完成后清理中间数据 |

---

## 8. 扩展性设计

### 7.3 HEhub 同态加密体系

HEhub 作为 Git 子模块集成到项目中，提供自研的 C++ 同态加密实现。

| 方案 | 安全假设 | 明文类型 | 自举 | 典型场景 |
|------|---------|---------|------|---------|
| **CKKS** | RLWE | 浮点数 (近似) | ❌ (Leveled) | 隐私机器学习 |
| **BGV** | RLWE | 整数 (精确) | ❌ (Leveled) | 隐私数据库查询 |
| **TFHE** | LWE | 布尔值/整数 | ✅ (FBS/FFBS) | 隐私电路计算 |

HEhub 的底层密码学原语包括：
- **RLWE**: Ring Learning With Errors 加密框架
- **RNS**: 残数系统分解大模数为小模数
- **NTT**: 数论变换实现 O(n log n) 多项式乘法
- **RGSW**: 用于密钥交换和自举的加密格式

---

**C++ 原生扩展**:
1. 在 `src/primihub/algorithm/` 下实现新算法
2. 在 `src/primihub/task/semantic/` 下添加对应的语义处理器
3. 在 `proto` 文件中添加新的消息类型
4. 注册到任务调度器

**Python SDK 扩展**:
1. 在 `python/primihub/FL/` 下实现新的 FL 算法
2. 使用 `@ph.context.register` 装饰器注册函数
3. 通过 `ph.context.Context` 传递参数

### 8.2 数据源扩展

在 `src/primihub/data_store/` 下添加新的数据驱动实现，需继承基础 Driver 接口：

```
data_store/
├── driver.h              # 基础驱动接口
├── csv/
│   ├── csv_driver.h
│   └── csv_driver.cc
├── sqlite/
├── mysql/
├── hdfs/
├── parquet/
└── image/
```
