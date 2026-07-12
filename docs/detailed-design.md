# PrimiHub 详细设计文档

## 1. 节点服务设计

### 1.1 节点启动流程

```
main()
  │
  ├─► 解析命令行参数
  │    --node_id, --config, --config_path
  │    --certificate, --key, --seed
  │
  ├─► 加载配置文件 (YAML)
  │    node: "node0"
  │    location: "primihub-node0"
  │    grpc_port: 50050
  │    use_tls: false
  │    meta_service: {mode: "grpc", ip: "...", port: 9099}
  │    datasets: [...]
  │
  ├─► 初始化日志系统 (glog)
  │    GLOG_logtostderr, GLOG_v
  │
  ├─► 初始化 gRPC 服务器
  │    ServerBuilder
  │    ├─► 注册 VMNode 服务
  │    ├─► 注册 DataSetService
  │    └─► 监听端口
  │
  ├─► 连接元数据服务
  │    ├─► 注册本节点信息
  │    ├─► 拉取已注册数据集
  │    └─► 建立协作关系
  │
  └─► 进入事件循环
       WaitForShutdown()
```

### 1.2 节点配置结构

```yaml
version: "1.0"
node: "node0"                    # 节点 ID
location: "primihub-node0"       # 节点位置标识
use_tls: false                   # 是否启用 TLS

grpc_port: 50050                 # gRPC 服务端口

# 代理服务器配置（用于网络受限环境）
proxy_server:
  mode: "grpc"
  ip: "127.0.0.1"
  port: 50050
  use_tls: false

# 元数据服务连接配置
meta_service:
  mode: "grpc"
  ip: "primihub-meta0"
  port: 9099
  use_tls: false

# 本地注册的数据集列表
datasets:
  - description: "psi_client_data"
    model: "csv"
    source: "data/client_e.csv"
  - description: "mpc_statistics_0"
    model: "csv"
    source: "data/mpc_test.csv"

# TLS 证书配置（可选）
# certificate:
#   root_ca: "data/cert/ca.crt"
#   key: "data/cert/node0.key"
#   cert: "data/cert/node0.crt"
```

### 1.3 节点间通信类结构

```
class NodeImpl : public VMNode::Service {
  // gRPC 服务实现
  Status SubmitTask(ServerContext*, PushTaskRequest*, PushTaskReply*);
  Status ExecuteTask(ServerContext*, PushTaskRequest*, PushTaskReply*);
  Status KillTask(ServerContext*, KillTaskRequest*, KillTaskResponse*);
  Status FetchTaskStatus(ServerContext*, TaskContext*, TaskStatusReply*);
  Status Send(ServerContext*, ServerReader<TaskRequest>*, TaskResponse*);
  Status Recv(ServerContext*, TaskRequest*, ServerWriter<TaskResponse>*);
  
private:
  Node* node_;                    // 节点状态
  map<string, Nodelet*> workers_; // 活跃 worker
  mutex mu_;                      // 并发控制
};

class Nodelet {
  // 任务执行单元
  void executeTask();
  void dispatchTask();
  void waitForTask();
  
private:
  string node_id_;
  string role_;
  Node* node_;                    // 所属节点
  TaskContext task_info_;         // 任务上下文
  unique_ptr<TaskExecutor> executor_;
};

class NodeStub {
  // 远程节点通信桩
  Status Send(const string& node_id, const TaskRequest& req);
  Status Recv(const string& node_id, TaskResponse* resp);
  Status ForwardSend(const string& dest, const TaskRequest& req);
};
```

---

## 2. 任务框架设计

### 2.1 任务模型

```
Task
├── task_type: TaskType           # PIR/PSI/MPC/TEE/ACTOR/NODE
├── name: string                  # 任务名称
├── language: Language            # PYTHON/CPP/PROTO
├── params: Params                # 任务参数
├── code: bytes                   # 任务代码
├── node_map: map<string, Node>   # 参与节点映射
├── party_datasets: map           # 参与方 -> 数据集
├── party_access_info: map        # 参与方 -> 访问信息
├── auxiliary_server: map         # 辅助服务
└── algorithm: Algorithm          # 算法类型
    ├── function_type: ADD/SUB/MUL/DIV/MAX/MIN/...
    └── operation_params
```

### 2.2 任务状态机

```
        ┌──────────┐
        │ PENDING  │
        └────┬─────┘
             │ SubmitTask
        ┌────▼─────┐
        │ RUNNING  │◄────────────────┐
        └────┬─────┘                 │
             │                       │
    ┌────────┼─────────┐             │
    │        │         │             │
┌───▼──┐ ┌──▼───┐ ┌───▼───┐       │
│SUCCESS│ │FAIL  │ │KILLED │       │
└───┬───┘ └───┬──┘ └───┬───┘       │
    │         │        │           │
    └─────────┼────────┘           │
              │                    │
        ┌─────▼──────┐             │
        │  FINISHED  ├─────────────┘
        └────────────┘  (可重新提交)
```

### 2.3 任务配置格式

PSI 任务配置示例：

```json
{
  "task_type": "PSI_TASK",
  "task_name": "psi_ecdh_task",
  "params": {
    "clientIndex": {
      "description": "客户端数据集选中的列索引",
      "type": "INT32",
      "value": [0]
    },
    "serverIndex": {
      "description": "服务端数据集选中的列索引",
      "type": "INT32",
      "value": [0]
    },
    "psiType": {
      "description": "INTERSECTION=0, DIFFERENCE=1",
      "type": "INT32",
      "value": 0
    },
    "psiTag": {
      "description": "ECDH=0, KKRT=1",
      "type": "INT32",
      "value": 0
    },
    "outputFullFilename": {
      "description": "客户端结果保存路径",
      "type": "STRING",
      "value": "data/result/psi_result.csv"
    }
  },
  "party_datasets": {
    "CLIENT": {"CLIENT": "psi_client_data"},
    "SERVER": {"SERVER": "psi_server_data"}
  }
}
```

---

## 3. 隐私集合求交 (PSI) 详细设计

### 3.1 PSI 协议对比

| 特性 | ECDH-PSI | KKRT-PSI | TEE-PSI |
|------|---------|---------|---------|
| 协议类型 | 公钥密码 | 对称密码 + OT | 硬件安全 |
| 性能 | 较慢 | 快速 | 最快 |
| 安全模型 | 半诚实 | 半诚实 | 恶意 |
| 数据集大小 | 小规模 | 大规模 | 大规模 |
| 需要密钥对 | 是 | 否 | 否 |

### 3.2 ECDH-PSI 协议流程

```
Client (拥有数据集 A)                    Server (拥有数据集 B)
         │                                      │
         │  生成密钥对 (sk_c, pk_c)              │
         │                                      │  生成密钥对 (sk_s, pk_s)
         │                                      │
         │  H(a) for each a ∈ A                 │
         │  E(H(a), sk_c)                       │
         │─────────────────────────────────────►│
         │                                      │  E(H(a), sk_s) for each
         │                                      │
         │◄─────────────────────────────────────│
         │                                      │
         │  E(H(b), sk_c) for each b ∈ B        │  H(b) for each b ∈ B
         │                                      │  E(H(b), sk_s)
         │                                      │
         │  计算双重加密：                        │
         │  E(E(H(a), sk_s), sk_c)              │
         │                                      │
         │  与 E(E(H(b), sk_c), sk_s) 比较       │
         │  匹配的即为交集                        │
```

### 3.3 PSI 核心类设计

```
class PsiExecutor {
public:
  virtual ~PsiExecutor() = default;
  virtual int execute(PartyConfig& config,
                      std::shared_ptr<DatasetService> service) = 0;
};

class EcdhPsiExecutor : public PsiExecutor {
  // ECDH 协议实现
  int execute(PartyConfig& config,
              std::shared_ptr<DatasetService> service) override;
private:
  void keyExchange();
  void encryptData();
  void computeIntersection();
  void saveResult();
};

class KkrtPsiExecutor : public PsiExecutor {
  // KKRT 协议实现
  int execute(PartyConfig& config,
              std::shared_ptr<DatasetService> service) override;
private:
  std::unique_ptr<osuCrypto::PsiReceiver> receiver_;
  std::unique_ptr<osuCrypto::PsiSender> sender_;
};
```

### 3.4 PSI 任务语义处理

```
class PSITask : public TaskBase {
  // 解析 PSI 任务配置
  int buildTask() override;
  int dispatch() override;
  int execute() override;
  
private:
  // 解析参数
  int parseParams(const PushTaskRequest& request);
  // 创建执行器
  std::unique_ptr<PsiExecutor> createExecutor(PsiTag tag);
  // 加载数据集
  std::shared_ptr<Dataset> loadDataset(const string& description);
};
```

---

## 4. 隐私信息检索 (PIR) 详细设计

### 4.1 PIR 协议流程

```
Client                               Server
  │                                     │
  │  ┌─ 生成查询 ─┐                     │
  │  │ 选择查询项 x │                    │
  │  │ 加密为 E(x) │                    │
  │  └──────┬─────┘                     │
  │         │                           │
  │  E(x)   │                           │
  │────────────────────────────────────►│
  │                                     │
  │         ┌─ 处理查询 ────┐           │
  │         │ 对每个记录 i： │           │
  │         │ 计算响应 R_i  │           │
  │         └──────┬───────┘           │
  │                │                    │
  │  {R_i}         │                    │
  │◄────────────────────────────────────│
  │                                     │
  │  ┌─ 解析结果 ─┐                     │
  │  │ 解密得到目标值 │                  │
  │  │ 服务器不知 x │                   │
  │  └─────────────┘                    │
```

### 4.2 PIR 数据索引结构

```
PIR 数据组织:

Database (N 条记录)
  │
  ├── ID Index (ID-PIR)
  │     └── HashMap<id, position>
  │
  └── Keyword Index (Keyword-PIR)
        └── Inverted Index<keyword, list<position>>

数据分片:
  ┌──────────────┬──────────────┬──────────────┐
  │  Shard 0     │  Shard 1     │  Shard 2     │
  │  记录 0-N/3  │  记录 N/3-N  │  记录 2N/3-N │
  └──────────────┴──────────────┴──────────────┘
  
每个分片可以并行处理以加速检索
```

---

## 5. 联邦学习 (FL) 详细设计

### 5.1 横向联邦学习流程

```
Server (聚合方)
    │
    │  ┌─ 初始化 ──────────────────────┐
    │  │ 分发初始模型参数               │
    │  └────────────┬──────────────────┘
    │               │
    │     ┌─────────┼─────────┐
    │     │         │         │
    ▼     ▼         ▼         ▼
  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
  │Client1│ │Client2│ │Client3│ │...  │   ← 多方本地训练
  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
     │       │       │       │
     └───────┴───────┴───────┘
               │
               │  ┌─ 安全聚合 ─┐
               │  │ 加密梯度    │
               │  │ 聚合更新    │
               │  └──────┬─────┘
               ▼
          ┌────────┐
          │ 更新模型 │              ← 多轮迭代
          └────────┘
```

### 5.2 纵向联邦学习流程

```
Party A (Host)                         Party B (Guest)
    │                                       │
    │  ┌─ PSI 对齐样本 ──┐                  │
    │  │ 找出共同样本 ID │                  │
    │  └─────────────────┘                  │
    │                                       │
    │  ┌─ 计算中间结果 ─┐                  │
    │  │ 本地特征计算    │                  │
    │  │ 加密后发送      │                  │
    │  └───────┬─────────┘                  │
    │           │                           │
    │  E(u_a)   │                           │
    │──────────────────────────────────────►│
    │                                       │
    │                           ┌─ 计算梯度 ┐│
    │                           │ 接收 E(u_a)│
    │                           │ 计算梯度 g │
    │                           │ 加密发送   │
    │                           └─────┬─────┘│
    │           │                     │      │
    │  E(g)      │◄────────────────────┘      │
    │◄────────────────────────────────────────│
    │               │                         │
    │  ┌─ 更新模型 ─┐                        │
    │  │ 解密梯度    │                        │
    │  │ 更新本地模型 │                        │
    │  └─────────────┘                        │
```

### 5.3 FL 算法模块类图

```
FL 算法类层次：

ModelBase (python/primihub/FL/utils/)
├── HFLModel
│   ├── HFLLinearRegression
│   ├── HFLLogisticRegression
│   ├── HFLNeuralNetwork
│   ├── HFLCNN
│   └── HFLXGBoost
│
└── VFLModel
    ├── VFLLinearRegression
    ├── VFLLogisticRegression
    └── VFLXGBoost

BaseModel 接口:
- train()          // 训练入口
- predict()        // 预测
- save_model()     // 保存模型
- load_model()     // 加载模型
- encrypt_grad()   // 加密梯度
- decrypt_grad()   // 解密梯度
- aggregate()      // 聚合更新
```

### 5.4 差分隐私集成

```
DP-SGD 训练流程:

for each epoch:
    for each batch:
        1. 计算梯度 ∇L(w)
        2. 裁剪梯度: ∇L ← ∇L / max(1, ||∇L||₂ / C)
        3. 添加噪声: ∇L ← ∇L + N(0, σ²C²I)
        4. 更新模型: w ← w - η·∇L
        
隐私预算追踪 (dp-accounting):
- Rényi DP 会计
- (ε, δ)-DP 保证
- 支持自适应隐私预算分配
```

---

## 6. 安全多方计算 (MPC) 详细设计

### 6.1 ABY3 协议架构

ABY3 是一个三方可选的安全计算协议，使用 Secret Sharing 技术：

```
三台服务器: S0, S1, S2

数据分片:
  原始值 x = x0 + x1 + x2 (mod 2^64)
  S0 持有 x0, x1
  S1 持有 x1, x2
  S2 持有 x2, x0

加法: 本地逐分量相加
  每个服务器本地计算 z_i = x_i + y_i

乘法: 需要交互
  使用 Beaver Triples 辅助计算
```

### 6.2 MPC 统计运算实现

```
class MpcStatisticsExecutor {
  int execute(PartyConfig& config,
              shared_ptr<DatasetService> service) override;

  // 统计算子
  void computeSum(const vector<vector<double>>& data, double* result);
  void computeAvg(const vector<vector<double>>& data, double* result);
  void computeMax(const vector<vector<double>>& data, double* result);
  void computeMin(const vector<vector<double>>& data, double* result);

  // 统计检验
  void computeTTest(const vector<double>& group1,
                    const vector<double>& group2,
                    TTestResult* result);
  void computeFTest(const vector<vector<double>>& groups,
                    FTestResult* result);
  void computeChiSquare(const vector<vector<int>>& contingency,
                        ChiSquareResult* result);

  // 回归与相关性
  void computeRegression(const vector<vector<double>>& X,
                         const vector<double>& y,
                         RegressionResult* result);
  void computeCorrelation(const vector<double>& x,
                          const vector<double>& y,
                          double* result);
};
```

### 6.3 ABY3 乘法协议

```
Beaver Triple 生成（预处理阶段）:
  生成随机三元组 (a, b, c) 满足 c = a * b
  分片后各方持有 (ai, bi, ci)

在线阶段（计算 x * y）:
  1. S0 计算 e = x0 - a0, f = y0 - b0
  2. S0 发送 e, f 给 S1, S2
  3. S1 计算 e = x1 - a1, f = y1 - b1  
  4. S2 计算 e = x2 - a2, f = y2 - b2
  5. 各方计算 z = c + a*f + b*e + e*f
```

---

## 7. 数据存储层设计

### 7.1 数据驱动架构

```
class DatasetService {
  // 数据集注册与发现
  void registerDataset(const DatasetMeta& meta);
  DatasetMeta getDataset(const string& description);
  vector<DatasetMeta> listDatasets();

private:
  map<string, DatasetMeta> local_datasets_;
  MetaServiceClient meta_client_;
};

class Driver {
  // 数据驱动基类
  virtual ~Driver() = default;
  virtual std::shared_ptr<DataMesh> read() = 0;
  virtual int write(std::shared_ptr<DataMesh>) = 0;
  virtual int64_t getRowCount() = 0;
  virtual int64_t getColumnCount() = 0;
};

class CsvDriver : public Driver { /* CSV 读取 */ };
class SqliteDriver : public Driver { /* SQLite 读取 */ };
class MySQLDriver : public Driver { /* MySQL 读取 */ };
class HdfsDriver : public Driver { /* HDFS 读取 */ };
class ParquetDriver : public Driver { /* Parquet 读取 */ };
class ImageDriver : public Driver { /* 图片读取 */ };

// 驱动工厂
class DriverFactory {
  static std::unique_ptr<Driver> create(const string& model,
                                        const string& source);
};
```

### 7.2 数据集元数据结构

```protobuf
message MetaInfo {
  string id;              // 全局唯一 ID
  string driver;          // 数据源类型 (csv/sqlite/mysql/...)
  string access_info;     // 访问路径
  string address;         // 存储位置
  Visibility visibility;  // PUBLIC / PRIVATE
  repeated DataTypeInfo data_type;  // 字段类型定义
}
```

---

## 8. pybind11 C++/Python 绑定设计

### 8.1 绑定模块

| 共享库 | 导出接口 |
|--------|---------|
| `linkcontext.so` | FL 上下文、模型训练接口 |
| `opt_paillier_c2py.so` | 优化的 Paillier 加密/解密 |
| `ph_secure_lib.so` | 安全计算基元 |

### 8.2 代码示例

```cpp
// pybind11 绑定示例 (linkcontext)
PYBIND11_MODULE(linkcontext, m) {
    m.def("init_context", &initContext, "Initialize FL context");
    m.def("train_model", &trainModel, "Train FL model");
    m.def("predict", &predict, "Make prediction");

    py::class_<FLContext>(m, "FLContext")
        .def(py::init<>())
        .def("set_param", &FLContext::setParam)
        .def("get_param", &FLContext::getParam);
}
```

---

## 9. SQL 安全校验设计

### 9.1 校验器架构

```
class SqlSecurityEngine {
  ValidationResult validate(const string& sql);
  
private:
  unique_ptr<SqlParser> parser_;
  vector<unique_ptr<Validator>> validators_;
};

// 校验器链
validators_ = {
  make_unique<JoinValidator>(),       // 表连接安全
  make_unique<FilterValidator>(),      // 过滤条件
  make_unique<AggregateValidator>(),   // 聚合函数
  make_unique<GroupByValidator>(),     // 分组
  make_unique<OrderByValidator>(),     // 排序
  make_unique<SubqueryValidator>(),    // 子查询
  make_unique<WindowValidator>()       // 窗口函数
};

class ValidationResult {
  bool is_valid_;        // 是否通过校验
  string message_;       // 提示信息
  SecurityLevel level_;  // 安全级别
};
```

### 9.2 安全校验策略

| 校验器 | 校验规则 | 违规处理 |
|--------|---------|---------|
| AggregateValidator | 禁止无限制的 COUNT(DISTINCT) | 拦截 SQL |
| FilterValidator | WHERE 条件必须包含分区键 | 添加默认过滤 |
| GroupByValidator | GROUP BY 列基数必须大于阈值 | 拒绝执行 |
| JoinValidator | 连接的关联键必须索引 | 警告并拦截 |
| OrderByValidator | ORDER BY 不能用于敏感列 | 移除排序列 |
| SubqueryValidator | 子查询嵌套层级限制 | 扁平化处理 |
| WindowValidator | 窗口函数范围限制 | 限制分区大小 |

---

## 10. 日志与监控

### 10.1 日志系统

```cpp
// C++ 日志 (glog)
// 通过 GLOG_v 环境变量控制日志级别:
// 0 = 最小日志, 7 = 最详细日志

LOG(INFO) << "Task submitted: " << task_id;     // 常规信息
LOG(WARNING) << "High memory usage";             // 警告
LOG(ERROR) << "Task execution failed";           // 错误
VLOG(3) << "Party config: " << config.DebugString();  // 调试
```

```python
# Python 日志 (loguru)
from loguru import logger

logger.info("Client initializing...")
logger.debug(f"Task config: {config}")
logger.error(f"gRPC connection failed: {e}")
```

### 10.2 健康检查

元数据服务提供的 HTTP 健康检查端点：

```
GET /health
Response: 200 OK (正常运行)
```

---

## 11. 关键数据结构

### 11.1 网络消息结构

```protobuf
message PushTaskRequest {
  bytes intended_worker_id = 1;  // 目标 Worker ID
  Task task = 2;                 // 任务描述
  int64 sequence_number = 3;     // 序列号（消息有序）
  int64 client_processed_up_to = 4;  // 客户端已处理到
  bytes submit_client_id = 5;    // 提交客户端 ID
  bool manual_launch = 6;        // 手动启动标志
}

message Node {
  bytes node_id = 1;     // 节点 ID
  bytes ip = 2;          // IP 地址
  int32 port = 3;        // gRPC 端口
  int32 data_port = 4;   // 数据端口
  bool use_tls = 5;      // 是否启用 TLS
  bytes role = 6;        // 节点角色
  repeated VirtualMachine vm = 7;  // 虚拟化信息
  int32 party_id = 8;    // 参与方 ID
}

message TaskContext {
  string task_id = 1;      // 任务 ID
  string job_id = 2;       // 作业 ID
  string request_id = 3;   // 请求 ID
  string sub_task_id = 4;  // 子任务 ID
}
```

## 12. MPC 协议体系

PrimiHub 实现了 **4 类 MPC 安全协议** + **3 类 PSI/PIR 协议** + **2 类同态加密**。

### 12.1 协议总览

```
┌──────────────────────────────────────────────────────────────┐
│                     PrimiHub 协议体系                          │
├──────────────────────────────────────────────────────────────┤
│  MPC 安全计算协议                                              │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐               │
│  │  ABY3    │  │  CryptFlow2  │  │  Falcon  │               │
│  │ (3方通用) │  │  (2方MaxPool)│  │ (3方 LeNet)│              │
│  └──────────┘  └──────────────┘  └──────────┘               │
├──────────────────────────────────────────────────────────────┤
│  PSI 隐私求交协议                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                    │
│  │ ECDH-PSI │  │ KKRT-PSI │  │ TEE-PSI  │                    │
│  └──────────┘  └──────────┘  └──────────┘                    │
├──────────────────────────────────────────────────────────────┤
│  PIR 隐私信息检索协议                                          │
│  ┌──────────┐  ┌─────────────┐  ┌──────────┐                 │
│  │ ID-PIR   │  │ Keyword-PIR │  │ PIR-ACC  │                 │
│  │          │  │  (APSI+SEAL) │  │ (GPU加速)│                 │
│  └──────────┘  └─────────────┘  └──────────┘                 │
├──────────────────────────────────────────────────────────────┤
│  同态加密                                                     │
│  ┌──────────┐  ┌──────────┐                                   │
│  │ Paillier │  │   CKKS   │                                   │
│  │ (加法同态)│  │(浮点近似) │                                   │
│  └──────────┘  └──────────┘                                   │
└──────────────────────────────────────────────────────────────┘
```

### 12.2 MPC 安全计算协议

#### 12.2.1 ABY3（核心协议）

基于 **3 方复制秘密共享**（Replicated Secret Sharing）的通用 MPC 框架，支持算术共享、布尔共享、定点数混合协议。

| 属性 | 说明 |
|------|------|
| **参与方** | 3 方 |
| **出处** | Mohassel & Rindal, CCS 2018 |
| **依赖** | `aby3` (PrimiHub fork) + `cryptoTools` |
| **核心类** | `MPCOperator` in `operator/aby3_operator.h` |
| **调度器** | `ABY3Scheduler` in `task/semantic/scheduler/aby3_scheduler.h` |
| **通信拓扑** | 环形拓扑（prev/next 双通道） |

**基础算子（11 个）**：

| 算子 | 方法 | 支持类型 |
|------|------|---------|
| ADD | `MPC_Add` / `MPC_Add_Const` | sf64 + si64 / 矩阵 + 标量 |
| SUB | `MPC_Sub` / `MPC_Sub_Const` | sf64 + si64 / 矩阵 + 标量 |
| MUL | `MPC_Mul` / `MPC_Mul_Const` | sf64 + si64 |
| DOT_MUL | `MPC_Dot_Mul` | 逐元素安全乘法 |
| DIV | `MPC_Div` | Goldschmidt 迭代算法 |
| CMP | `MPC_Compare` | MSB 电路（Kogge-Stone Adder） |
| ABS | `MPC_Abs` | 分段函数求绝对值 |
| DReLu | `MPC_DReLu` | 符号判断（>0 返回 1） |
| Pow | `MPC_Pow` | 2^rank（值 > 0.5） |
| Pow2 | `MPC_Pow2` | 2^(-rank)（值 < 0.5） |
| QuoDertermine | `MPC_QuoDertermine` | 商符号确定 |

**高级统计算子（9 个）**：

| 算子 | 实现类 | 文件 |
|------|--------|------|
| SUM | `MPCSumOrAvg(SUM)` | `executor/statistics.h` |
| AVG | `MPCSumOrAvg(AVG)` | `executor/statistics.h` |
| MAX | `MPCMinOrMax(MAX)` | `executor/statistics.h` |
| MIN | `MPCMinOrMax(MIN)` | `executor/statistics.h` |
| T_TEST | `MPCTTest` | `executor/statistical_tests.h` |
| F_TEST | `MPCFTest` | `executor/statistical_tests.h` |
| CHI_SQUARE | `MPCChiSquareTest` | `executor/statistical_tests.h` |
| REGRESSION | `MPCRegression` | `executor/statistical_tests.h` |
| CORRELATION | `MPCCorrelation` | `executor/statistical_tests.h` |

**ABY3 乘法协议（Beaver Triple）**：
```
离线阶段：生成随机三元组 (a, b, c = a·b) 的分片
在线阶段（计算 x·y）：
  1. 各参与方计算 e = x_i - a_i, f = y_i - b_i
  2. 公开 e, f 给其他方
  3. 各参与方计算 z = c + a·f + b·e + e·f
```

#### 12.2.2 CryptFlow2（2 方 MPC）

| 属性 | 说明 |
|------|------|
| **用途** | MaxPool 神经网络算子 |
| **参与方** | 2 方 |
| **依赖** | `cryptflow2/`（外部依赖） |
| **核心类** | `MaxPoolExecutor` in `algorithm/cryptflow2_maxpool.h` |
| **调度器** | `CRYPTFLOW2Scheduler` |

#### 12.2.3 Falcon（3 方 MPC NN）

| 属性 | 说明 |
|------|------|
| **用途** | LeNet 神经网络安全推理 |
| **参与方** | 3 方 |
| **依赖** | `falcon-public/`（外部依赖） |
| **核心类** | `FalconLenetExecutor` in `algorithm/falcon_lenet.h` |
| **调度器** | `FalconScheduler` |

### 12.3 PSI 隐私集合求交协议

| 协议 | 类型 | 参与方 | 核心文件 |
|------|------|--------|---------|
| **ECDH-PSI** | 椭圆曲线公钥 | 2 方 | `kernel/psi/operator/ecdh_psi.h` |
| **KKRT-PSI** | OT 基对称密钥 | 2 方 | `kernel/psi/operator/kkrt_psi.h` |
| **TEE-PSI** | Intel SGX 硬件 | 2 方 | `kernel/psi/operator/tee_psi.h` |

- 统一接口：`BasePsiOperator` → `EcdhPsiOperator` / `KkrtPsiOperator` / `TeePsiOperator`
- 工厂调度：`kernel/psi/operator/factory.h`（根据 `PsiTag` 分发）
- 依赖：`libPSI`（KKRT）、`OpenMined PSI`（ECDH）、`SGX SDK`（TEE）
- 支持模式：`INTERSECTION`（交集）、`DIFFERENCE`（差集）

**ECDH-PSI 协议流程**：
```
Client (A)                          Server (B)
   │                                    │
   │  H(a) for each a ∈ A                │
   │  E(H(a), sk_c)                      │
   │────────────────────────────────────►│
   │                                    │  E(H(a), sk_s) for each
   │◄────────────────────────────────────│
   │                                    │
   │  双重加密后比较匹配即为交集            │
```

### 12.4 PIR 隐私信息检索协议

| 协议 | 技术路线 | 核心文件 |
|------|---------|---------|
| **ID-PIR** | 基于标识符 | `kernel/pir/operator/id_pir.h` |
| **Keyword-PIR** | APSI + SEAL 同态加密 | `kernel/pir/operator/keyword_pir_impl/` |
| **PIR-ACC** | GPU 加速 | `kernel/pir-acc/`（CUDA） |

- 依赖：`Microsoft APSI`、`Microsoft SEAL 4.0`、`Kuku`（Cuckoo Hashing）
- 工厂调度：`kernel/pir/operator/factory.h`

### 12.5 同态加密

| 方案 | 类型 | 语言 | 库 | 路径 |
|------|------|------|-----|------|
| **BGV** | 整数同态加密 | C++ | **HEhub** | `hehub/src/fhe/bgv/` |
| **CKKS** | 浮点近似同态 | C++ | **HEhub** | `hehub/src/fhe/ckks/` |
| **TFHE** | 函数自举 | C++ | **HEhub** | `hehub/src/fhe/tfhe/` |
| **Paillier** | 加法同态（GMP 优化） | C++ | 自研 | `algorithm/opt_paillier/` |
| **Paillier** | 加法同态 | Python | phe | `python/primihub/FL/crypto/paillier.py` |
| **CKKS** | 浮点近似同态 | Python | TenSEAL | `python/primihub/FL/crypto/ckks.py` |

### 12.6 HEhub 同态加密库

HEhub 是 PrimiHub 子项目，提供自研的 C++ 同态加密实现，支持 **BGV / CKKS / TFHE** 三种 FHE 方案。

#### 12.6.1 整体架构

```
hehub/
├── src/
│   ├── fhe/                      # FHE 方案实现
│   │   ├── bgv/                  # BGV 整数同态加密
│   │   │   ├── bgv.h             # 参数/明文/密文类型 + API
│   │   │   ├── basics.cpp        # 编码/加密/解密
│   │   │   └── mod_switch.cpp    # 模切换运算
│   │   ├── ckks/                 # CKKS 浮点同态加密
│   │   │   ├── ckks.h            # 参数/明文/密文类型 + API
│   │   │   ├── basics.cpp        # 编码/加密/解密
│   │   │   ├── arith.cpp         # 加/减/乘法运算
│   │   │   └── rescaling.cpp     # 缩放(rescale)运算
│   │   ├── tfhe/                 # TFHE 函数自举
│   │   │   └── func_boot.h       # FBS/FFBS 函数自举 API
│   │   ├── primitives/           # 密码学原语
│   │   │   ├── rlwe.h            # RLWE 加密: 参数、密钥、加密/解密
│   │   │   ├── lwe.h             # LWE 加密类型定义
│   │   │   ├── rgsw.h            # RGSW 加密: 加密 + 外积
│   │   │   └── keys.h            # 密钥类型: 重线性化/共轭/旋转密钥
│   │   └── common/               # 公共工具
│   │       ├── bigint.h          # 无限精度大整数
│   │       ├── mod_arith.h       # 模算术 (Barrett/Montgomery/Harvey)
│   │       ├── ntt.h             # 数论变换 (NTT/INTT)
│   │       ├── rns.h             # 残数系统 (RNS 多项式)
│   │       ├── permutation.h     # 自同态 (involution/cycle)
│   │       ├── sampling.h        # 随机采样 (ternary/gaussian/uniform)
│   │       ├── primelists.h      # 预计算质数表
│   │       ├── allocator.h       # 智能内存分配器
│   │       └── type_defs.h       # 基础类型定义
│   └── circuits/                 # 高级电路
│       ├── linear_algebra.h      # CKKS 加密矩阵-向量乘法
│       ├── ckks_boot.h           # CKKS 自举 (预留)
│       ├── cc_non_poly.h         # 非多项式函数 (预留)
│       └── fp_non_poly.h         # 定点数非多项式 (预留)
├── tests/                        # 测试 (Catch2)
└── bench/                        # 基准测试
```

#### 12.6.2 CKKS 方案

CKKS（Cheon-Kim-Kim-Song）是支持浮点数近似计算的同态加密方案。

| 功能 | API | 说明 |
|------|-----|------|
| 参数生成 | `create_params(dim, scaling_bits)` | 自动确定模数和缩放因子 |
| 编码 | `simd_encode(data, params)` | 将 double/complex 编码为明文 |
| 解码 | `simd_decode(pt)` | 将明文解回 double/complex |
| 加密 | `encrypt(pt, sk)` | 明文 → 密文 |
| 解密 | `decrypt(ct, sk)` | 密文 → 明文 |
| 加法 | `add(ct1, ct2)` / `add_plain(ct, pt)` | 密文+密文 / 密文+明文 |
| 减法 | `sub(ct1, ct2)` / `sub_plain(ct, pt)` | 密文-密文 / 密文-明文 |
| 明文乘法 | `mult_plain(ct, pt)` | 密文 × 明文 |
| 密文乘法 | `mult_low_level` → `relinearize` → `rescale_inplace` | 3 步: 低阶乘 → 重线性化 → 缩放 |
| 重线性化 | `relinearize(quad_ct, relin_key)` | 缩回 2 组件密文 |
| 共轭 | `conjugate(ct, conj_key)` | 复数共轭 |
| 旋转 | `rotate(ct, rot_key, step)` | 槽间旋转 |
| 缩放 | `rescale_inplace(ct)` | 移除一个素数降低噪声 |

**CKKS 参数体系**：

```
维度 N     ∝ 安全性 (典型值: 4096/8192)
槽数 N/2   = 每个密文能打包的实数/复数个数
模数链      = 比特长度列表 [60, 40, 40, ...]
缩放因子   = 编码精度 = 2^scaling_bits
附加模     = 用于重线性化的额外模数
```

**CKKS 乘法过程**：

```
输入: ct1, ct2 (各 2 个多项式)
  1. mult_low_level: 计算 3 个多项式 (二次密文)
  2. relinearize: 用重线性化密钥将 3→2 多项式
  3. rescale: 移除一个素数，降低缩放因子
输出: ct_prod (2 个多项式)
```

#### 12.6.3 BGV 方案

BGV（Brakerski-Gentry-Vaikuntanathan）是支持整数运算的同态加密方案。

| 功能 | API | 说明 |
|------|-----|------|
| 编码 | `simd_encode(data, p)` | 将 u64 向量编码为明文 |
| 解码 | `simd_decode(pt)` | 将明文解码为 u64 向量 |
| 加密 | `encrypt(pt, sk)` | 明文 → 密文 |
| 解密 | `decrypt(ct, sk)` | 密文 → 明文 |
| 加法 | `add` / `add_plain` | 密文/明文加法 |
| 减法 | `sub` / `sub_plain` | 密文/明文减法 |
| 明文乘法 | `mult_plain` | 密文 × 明文 |
| 密文乘法 | `mult_low_level` → `relinearize` | 与 CKKS 类似 |
| 模切换 | `mod_switch_inplace` | 降低密文模数 |

BGV 与 CKKS 共享相同的底层原语（RLWE、RNS、NTT），区别在于：
- BGV 使用整数明文模数 p，CKKS 使用浮点缩放因子
- BGV 解密要求噪声小于 p/2，CKKS 容忍近似误差
- BGV 支持精确运算，CKKS 支持高精度近似

#### 12.6.4 TFHE 方案

TFHE（Chillotti et al.）通过**函数自举**实现任意布尔/整数电路的 FHE。

| 功能 | API | 说明 |
|------|-----|------|
| 函数自举 | `functional_bootstrap(ct, lut_poly, boot_keys)` | 用查找表计算任意函数 |
| 冗余 MSB | `get_redundant_msb(ct, boot_keys)` | 提取盲旋转的冗余比特 |
| 完全函数自举 | `fully_functional_bootstrap(ct, lut_poly, boot_keys)` | 避免负周期的 FFBS |

**FBS 原理**：
```
输入 LWE 密文 (a, b) + 查找表多项式 LUT(X)
1. 盲旋转: 计算 LUT(X) · X^{(a·s + b)} 的加密
2. 提取: 从 RLWE 密文中提取目标槽的 LWE 密文
3. 输出: LUT[round(b + a·s)] 的加密
```

#### 12.6.5 底层原语

| 原语 | 文件 | 功能 |
|------|------|------|
| **RLWE** | `primitives/rlwe.h` | RLWE 加密/解密、密钥生成、加/减/乘 |
| **RGSW** | `primitives/rgsw.h` | RGSW 加密（用于密钥交换/自举） |
| **密钥** | `primitives/keys.h` | `RlweKsk`(重线性化)、`RotKey`(旋转) |
| **NTT** | `common/ntt.h` | 数论变换(多项式乘法 O(n log n)) |
| **RNS** | `common/rns.h` | 残数系统(大整数拆分为多个小模数) |
| **BigInt** | `common/bigint.h` | 无限精度大整数(CRT 组合) |
| **模算术** | `common/mod_arith.h` | Barrett/Montgomery/Harvey 模约简 |
| **采样** | `common/sampling.h` | 三元/高斯/均匀随机多项式采样 |
| **置换** | `common/permutation.h` | 自同态(involution/cycle) |
| **素数** | `common/primelists.h` | 预计算 60bit 以内素数列表 |

#### 12.6.6 高级电路

| 电路 | 文件 | 功能 |
|------|------|------|
| **矩阵-向量乘** | `circuits/linear_algebra.h` | 加密向量 × 明文矩阵的私密机器学习推理 |

**矩阵-向量乘法算法**（对角线打包）：
```
输入: 加密向量 ct_vec + 明文矩阵 mat (height × width)
1. 将矩阵按对角线打包: diag[i][j] = mat[j][(j+width-i) % width]
2. 对每个对角线 i:
   a. 加密旋转向量: ct_vec_rot = rotate(ct_vec, i)
   b. 明文乘: ct_prod = mult_plain(ct_vec_rot, encode(diag[i]))
   c. 累加: ct_acc = add(ct_acc, ct_prod)
3. 输出: ct_acc (加密结果向量)

### 12.6 协议调度机制

```
scheduler/factory.h:
  ACTOR_TASK ── code == "maxpool"  ──► CRYPTFLOW2Scheduler (2-party)
             ├── code == "lenet"    ──► FalconScheduler (3-party)  
             └── else               ──► ABY3Scheduler (3-party)
  PSI_TASK   ── psi_tag            ──► ECDH/KKRT/TEE PSI
  PIR_TASK   ── pir_type           ──► ID_PIR / KEY_PIR
  TEE_TASK   ──                    ──► TEEScheduler
  PYTHON     ──                    ──► FLScheduler (Federated Learning)
```

### 12.7 外部依赖库

| 依赖 | 用途 | 协议/方案 |
|------|------|-----------|
| `com_github_ladnir_aby3` | 3 方安全计算框架 | ABY3 |
| `ladnir_cryptoTools` | 密码学工具 | ABY3 |
| `osu_libote` | 不经意传输 | ABY3 / PSI |
| `osu_libpsi` | PSI 协议 | KKRT-PSI |
| `org_openmined_psi` | PSI 协议 | ECDH-PSI |
| `mircrosoft_apsi` | 隐私信息检索 | Keyword-PIR |
| `com_github_primihub_seal_40` | 同态加密库 | CKKS |
| `com_github_gmp` | 大数运算 | Paillier |
| **hehub** (本地子模块) | 自研 C++ 同态加密库 | BGV / CKKS / TFHE |

**HEhub vs. 其他 HE 方案对比**：

| 特性 | HEhub (C++) | TenSEAL (Python) | phe (Python) |
|------|-------------|-------------------|--------------|
| 方案 | BGV, CKKS, TFHE | CKKS | Paillier |
| 类型 | 自研 C++ 库 | SEAL Python 绑定 | python-paillier 绑定 |
| 性能 | 原生 C++ (LTO/OMP) | C++ 后端 | Python |
| 安装 | CMake 编译 | pip install | pip install |
| 集成方式 | Git 子模块 | pip 包 | pip 包 |
| 用途 | 原生 C++ 同态运算 | FL Python 算法 | FL Python 算法 |

### 12.8 算法类型定义

```protobuf
message Algorithm {
  enum Type {
    Arithmetic = 0;
    Statistics = 1;
  }
  enum StatisticsOpType {
    MAX = 0;  MIN = 1;  AVG = 2;  SUM = 3;
    T_TEST = 4;  F_TEST = 5;  CHI_SQUARE_TEST = 6;
    REGRESSION = 7;  CORRELATION = 8;
  }
  enum ArithmeticOpType {
    ADD = 0;  SUB = 1;  MUL = 2;  DIV = 3;  CMP = 4;
  }
  Type function_type = 1;
}
```
