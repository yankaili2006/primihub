# PrimiHub 联邦学习架构设计文档

## 1. 概述

### 1.1 什么是联邦学习

联邦学习（Federated Learning，FL）是一种分布式机器学习范式，多个参与方在不共享原始数据的前提下，协作训练机器学习模型。核心原则是"数据不动模型动"——模型在各参与方本地训练，仅交换模型参数或梯度。

### 1.2 PrimiHub FL 定位

PrimiHub FL 是平台隐私计算能力的重要组成部分，覆盖横向联邦（HFL）和纵向联邦（VFL）两种模式，提供线性回归、逻辑回归、XGBoost、神经网络、大模型微调等主流算法，并集成密码学原语（Paillier、CKKS）和差分隐私（DPSGD）保护。

---

## 2. 总体架构

### 2.1 系统层次

```
┌──────────────────────────────────────────────────────────────┐
│                    应用层                                      │
│  JSON 任务配置  │  Python SDK  │  CLI 命令行                   │
├──────────────────────────────────────────────────────────────┤
│                    任务调度层 (C++)                            │
│  FLScheduler ──► FLTask ──► pybind11 嵌入 Python 解释器      │
├──────────────────────────────────────────────────────────────┤
│                    FL 算法层 (Python)                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ HFL      │ │ VFL      │ │ XGBoost  │ │ ChatGLM       │  │
│  │ LR/LogReg│ │ LR/LogReg│ │ (纵向)   │ │ (大模型微调)   │  │
│  │ NN/CNN   │ │          │ │          │ │               │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                    密码学层                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Paillier │ │   CKKS   │ │ DPSGD    │ │  差分隐私      │  │
│  │ (python) │ │(TenSEAL) │ │(Opacus)  │ │(dp-accounting) │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                    通信层                                      │
│  ┌────────────┐ ┌────────────┐ ┌──────────────────────────┐  │
│  │ GrpcClient │ │MultiGrpc   │ │   linkcontext (pybind11) │  │
│  │ (1对1)     │ │Clients(N对1│ │   C++ gRPC 通道          │  │
│  └────────────┘ └────────────┘ └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 部署架构

```
┌─── 数据提供方 A ───┐    ┌─── 数据提供方 B ───┐    ┌─── 协调方 ───────┐
│                    │    │                    │    │                   │
│  ┌────────────┐   │    │  ┌────────────┐   │    │  ┌─────────────┐  │
│  │ primihub   │   │    │  │ primihub   │   │    │  │ primihub    │  │
│  │ -node      │   │    │  │ -node      │   │    │  │ -node       │  │
│  └─────┬──────┘   │    │  └─────┬──────┘   │    │  └──────┬──────┘  │
│        │          │    │        │          │    │         │         │
│  ┌─────▼──────┐   │    │  ┌─────▼──────┐   │    │  ┌──────▼──────┐ │
│  │ FL Client  │   │    │  │ FL Server  │   │    │  │ Coordinator│ │
│  │ (HFL模式)  │   │    │  │ (聚合方)   │   │    │  │ (VFL模式)  │ │
│  └────────────┘   │    │  └────────────┘   │    │  └─────────────┘  │
│                    │    │                    │    │                   │
│  本地数据 + 本地   │    │  聚合全局模型       │    │  密钥生成/分发   │
│  训练 + 梯度加密   │    │                    │    │  噪声管理        │
└────────────────────┘    └────────────────────┘    └───────────────────┘
```

### 2.3 HFL 架构

```
       Server (聚合方)
           │
    ┌──────┼──────┐
    │      │      │
    ▼      ▼      ▼
 ┌────┐ ┌────┐ ┌────┐
 │C1  │ │C2  │ │C3  │  ← Client (数据持有方)
 │数据│ │数据│ │数据│
 └────┘ └────┘ └────┘

训练流程:
  1. Server 初始化全局模型 θ₀
  2. 广播 θₜ 给所有 Client
  3. 每个 Client 本地训练: θₜⁱ = LocalUpdate(θₜ, Dᵢ)
  4. Client 发送 θₜⁱ (或梯度) 给 Server
  5. Server 聚合: θₜ₊₁ = Σ(nᵢ/N) · θₜⁱ
  6. 重复 2-5 直到收敛

隐私保护模式:
  Plaintext  — 直接传输明文梯度 (FedAvg)
  DPSGD      — 梯度裁剪 + 高斯噪声 + 隐私预算追踪
  Paillier   — 客户端加密梯度, 服务端密文聚合
```

### 2.4 VFL 架构

```
     Coordinator (协调方)
         ▲      │
         │      │
    ┌────┼──────┼────┐
    │    │      │    │
    ▼    ▼      ▼    ▼
 ┌────┐         ┌────┐
 │Guest│         │Host│
 │特征A│         │特征B│
 │无标签│         │有标签│
 └────┘         └────┘

训练流程:
  1. Coordinator 生成 CKKS 密钥对
  2. PSI 样本对齐 (找出共同样本)
  3. Guest 计算中间结果 z_guest = X_guest · W_guest
  4. Host 计算: z = X_host · W_host + b + z_guest
  5. Host 计算误差 error = z - y
  6. Guest 和 Host 各自计算梯度并更新本地模型
  7. Coordinator 定期解密/重加密管理噪声

隐私保护模式:
  Plaintext  — 明文传输中间结果
  CKKS       — 加密传输 (TenSEAL 近似同态加密)
```

---

## 3. 模块划分

### 3.1 C++ 基础设施

| 模块 | 路径 | 职责 |
|------|------|------|
| FLScheduler | `task/semantic/scheduler/fl_scheduler.h/cc` | 多节点 FL 任务分发 |
| FLTask | `task/semantic/fl_task.h/cc` | FL 任务执行 (pybind11 嵌入 Python) |
| GrpcClient | 底层通信 | gRPC 通道封装 |

### 3.2 Python 算法层

| 子模块 | 路径 | 算法 |
|--------|------|------|
| linear_regression | `FL/linear_regression/` | HFL/VFL 线性回归 |
| logistic_regression | `FL/logistic_regression/` | HFL/VFL 逻辑回归 |
| neural_network | `FL/neural_network/` | HFL MLP/CNN |
| xgboost | `FL/xgboost/` | VFL XGBoost |
| chatglm | `FL/chatglm/` | HFL ChatGLM 微调 |

### 3.3 Python 支持层

| 子模块 | 路径 | 功能 |
|--------|------|------|
| preprocessing | `FL/preprocessing/` | 17 个 sklearn 兼容预处理器 |
| stats | `FL/stats/` | 10 个统计函数 + 假设检验 |
| crypto | `FL/crypto/` | Paillier + CKKS 封装 |
| psi | `FL/psi/` | 样本对齐 |
| metrics | `FL/metrics/` | 回归/分类评估指标 |
| utils | `FL/utils/` | 通信、数据加载、文件 I/O |

---

## 4. 任务执行流程

### 4.1 FL 任务提交

```
CLI/ Python SDK
      │
      │ JSON config / Python code
      ▼
primihub-node (gRPC Server)
      │
      ▼
FLScheduler (C++)
      │
      ├──► 解析参与方角色 (client/server/guest/host/coordinator)
      ├──► 通过 gRPC 分发任务到各节点
      │
      ▼
每个节点上的 FLTask (C++)
      │
      ├──► 启动 Python 解释器 (pybind11)
      ├──► 根据 model_map.json 加载对应 Python 类
      ├──► 调用 BaseModel.run()
      │
      ▼
Python 算法执行
```

### 4.2 HFL 训练流程

```
Client (数据持有方)                     Server (聚合方)
      │                                      │
      │  1. 初始化:                           │
      │  - 建立 gRPC 通道                     │
      │  - 发送样本数 nᵢ                     │
      │                                      │
      │  2. 全局迭代:                         │
      │     for epoch in range(global_epoch): │
      │       │                               │
      │       │  local_epoch 次:              │
      │       │  for batch in DataLoader:     │
      │       │    model.fit(batch)            │
      │       │                               │
      │       │  θᵢ ─────────►                │  聚合 θ = avg(θᵢ)
      │       │  ◄────────── θ                │
      │       │                               │
      │  3. 训练完成:                          │
      │  - 保存模型                           │
      │  - 计算 ε (DPSGD)                     │
```

### 4.3 VFL 训练流程

```
Guest                              Host                        Coordinator
  │                                  │                             │
  │  1. PSI 样本对齐 ◄───────────────►                             │
  │                                  │                             │
  │  2. 生成 CKKS 密钥 ◄────────────────────────────────────────── │
  │                                  │                             │
  │  3.  for epoch:                   │                             │
  │     z_g = X_g · W_g              │                             │
  │     E(z_g) ──────────►            │                             │
  │                                  │  z = X_h · W_h + b + E(z_g) │
  │                                  │  error = z - y              │
  │                                  │  dw_h, db = grad()          │
  │                                  │  W_h -= lr · dw_h           │
  │     ◄────────── E(error) ────────│                             │
  │     dw_g = grad()                │                             │
  │     W_g -= lr · dw_g             │                             │
  │                                  │                             │
  │  4. 每 max_iter 轮:               │                             │
  │     ◄────────── 解密/重加密 ─────│                             │
  │                                  │                             │
  │  5. 保存模型                      │                             │
```

---

## 5. 模型注册与发现

`model_map.json` 是 FL 算法的中心注册表：

```json
{
  "HFL_linear_regression": {
    "client": "primihub.FL.linear_regression.hfl_client.LinearRegressionClient",
    "server": "primihub.FL.linear_regression.hfl_server.LinearRegressionServer"
  },
  "VFL_linear_regression": {
    "guest": "primihub.FL.linear_regression.vfl_guest.LinearRegressionGuest",
    "host": "primihub.FL.linear_regression.vfl_host.LinearRegressionHost",
    "coordinator": "primihub.FL.linear_regression.vfl_coordinator.LinearRegressionCoordinator"
  }
}
```

---

## 6. 通信模型

### 6.1 GrpcClient (1 对 1)

基于 C++ `linkcontext` 的 pybind11 封装：
- 每个 FL 参与方维护 2 个通道（send + recv）
- 调用 `channel.send(key, pickle.dumps(val))` 传输序列化对象

### 6.2 MultiGrpcClients (1 对 N)

聚合方向多个客户端并行通信：
- `send_all(key, val)` — 广播
- `recv_all(key)` — 收集所有客户端结果
- `send_selected / recv_selected` — 选择性通信

---

## 7. 密码学集成

| 密码学方案 | 封装位置 | 使用场景 |
|-----------|---------|---------|
| Paillier (phe) | `FL/crypto/paillier.py` | HFL 线性/逻辑回归加密聚合 |
| CKKS (TenSEAL) | `FL/crypto/ckks.py` | VFL 线性/逻辑回归加密计算 |
| Optimized-Paillier (C++) | `primitive/opt_paillier_c2py_warpper.py` | XGBoost 梯度加密 |
| DPSGD | `opacus + dp-accounting` | HFL 差分隐私训练 |

---

## 8. 技术栈

| 层级 | 技术 |
|------|------|
| 核心框架 | C++17 (gRPC + pybind11) |
| 算法实现 | Python 3.8+ |
| 深度学习 | PyTorch + Opacus |
| 梯度提升 | XGBoost + Ray |
| 同态加密 | phe (Paillier) + TenSEAL (CKKS) |
| 通信 | gRPC + linkcontext (C++ pybind11) |
| 模型注册 | JSON 配置文件 (model_map.json) |
