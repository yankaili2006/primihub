# PrimiHub 联邦学习详细设计文档

## 1. 基础框架

### 1.1 BaseModel 抽象基类

所有 FL 算法的统一入口，定义在 `FL/utils/base.py`。

```python
class BaseModel(metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.roles = kwargs['roles']               # 角色映射
        self.common_params = kwargs['common_params'] # 全局参数
        self.role_params = kwargs['role_params']     # 角色本地参数
        self.node_info = kwargs['node_info']         # 节点信息
        self.task_info = kwargs['task_info']         # 任务信息

    @abstractmethod
    def run(self):
        pass  # 算法执行入口
```

`role_params` 结构示例：
```json
{
  "self_name": "node0",       // 自身节点名
  "self_role": "client",      // 自身角色
  "others_role": ["server"],  // 对端角色名
  "data": {"name": "/path/to/data.csv", "type": "csv"},
  "model_path": "data/result/model.pkl",
  "metric_path": "data/result/metrics.json"
}
```

### 1.2 通信封装

#### GrpcClient（1 对 1）

```python
class GrpcClient:
    def __init__(self, local_party, remote_party, node_info, task_info):
        # 通过 linkcontext (C++ pybind11) 创建 gRPC 通道
        self.link_context = linkcontext.LinkFactory.createLinkContext(GRPC)
        # 本地通道 (接收)
        self.recv_channel = self.link_context.getChannel(Node(local_ip, port, ...))
        # 远程通道 (发送)
        self.send_channel = self.link_context.getChannel(Node(remote_ip, port, ...))

    def send(self, key, val):
        self.send_channel.send(key, pickle.dumps(val))

    def recv(self, key):
        return pickle.loads(self.recv_channel.recv(key))
```

#### MultiGrpcClients（1 对 N）

```python
class MultiGrpcClients:
    def __init__(self, local_party, remote_parties, node_info, task_info):
        self.Clients = {party: GrpcClient(...) for party in remote_parties}

    def send_all(self, key, val):     # 广播
    def recv_all(self, key):          # 收集所有
    def send_selected(self, key, val, selected):   # 选择发送
    def recv_selected(self, key, selected):        # 选择接收
```

---

## 2. HFL 线性回归

### 2.1 类层次

```
LinearRegression (base.py)
  ├── Plaintext_Client (hfl_client.py)
  │     └── DPSGD_Client
  │     └── Paillier_Client
  └── Plaintext_DPSGD_Server (hfl_server.py)
        └── Paillier_Server
```

### 2.2 本地模型 (base.py)

```python
class LinearRegression:
    def __init__(self, x, learning_rate=0.2, alpha=0.0001):
        self.learning_rate = learning_rate
        self.alpha = alpha              # L2 正则化系数
        self.weight = np.zeros(x.shape[1])
        self.bias = np.zeros(1)

    def fit(self, x, y):
        error = self.predict(x) - y
        dw = x.T.dot(error) / x.shape[0] + self.alpha * self.weight
        db = error.mean(axis=0, keepdims=True)
        self.weight -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, x):
        return x.dot(self.weight) + self.bias

    def get_theta(self):      # 返回 [bias, weight...]
        return np.concatenate([self.bias.ravel(), self.weight.ravel()])

    def set_theta(self, theta):
        self.bias = theta[0]
        self.weight = theta[1:]
```

### 2.3 Client 训练逻辑 (hfl_client.py)

```python
class Plaintext_Client:
    def __init__(self, x, learning_rate, alpha, server_channel):
        self.model = LinearRegression(x, learning_rate, alpha)
        self.server_channel = server_channel
        self.num_examples = x.shape[0]
        self.send_params()

    def train(self):
        # 发送本地模型 → 接收聚合模型 (FedAvg)
        self.server_channel.send("client_model", self.model.get_theta())
        self.model.set_theta(self.server_channel.recv("server_model"))
```

### 2.4 Server 聚合逻辑 (hfl_server.py)

```python
class Plaintext_DPSGD_Server:
    def __init__(self, alpha, client_channel):
        self.alpha = alpha
        self.client_channel = client_channel
        self.num_examples_weights = self.client_channel.recv_all('num_examples')

    def client_model_aggregate(self):
        client_models = self.client_channel.recv_all("client_model")
        # 加权平均: θ = Σ(n_i/N) · θ_i
        self.theta = np.average(client_models,
                                weights=self.num_examples_weights,
                                axis=0)

    def server_model_broadcast(self):
        self.client_channel.send_all("server_model", self.theta)
```

### 2.5 Paillier 模式数据流

```
Server:                              Client:
  │                                      │
  │  ┌─ 生成 Paillier 密钥 ─┐            │
  │  │ pub_key, priv_key    │            │
  │  └──────────┬───────────┘            │
  │             │                        │
  │  pub_key ──────────────────────────► │
  │             │                        │  ┌─ 加密模型 ─┐
  │             │                        │  │ E(θ)      │
  │             │                        │  └─────┬─────┘
  │             │  E(θ) ◄─────────────── │        │
  │  ┌─ 密文聚合 ─┐                      │        │
  │  │ Σ E(θ_i)  │                      │        │
  │  │ 均值取整   │                      │        │
  │  └─────┬─────┘                      │        │
  │        │                            │        │
  │  ┌─ 解密 ─┐                         │        │
  │  │ D(E(θ))│                         │        │
  │  └────┬───┘                         │        │
  │       │                             │        │
  │  明文 θ ──────────────────────────► │        │
  │       │                             │ 设置本地模型
```

### 2.6 DPSGD 模式

```python
class LinearRegression_DPSGD:
    def __init__(self, x, lr, alpha, noise_multiplier, l2_norm_clip, secure_mode):
        # 梯度裁剪 + 噪声注入
        self.noise_multiplier = noise_multiplier
        self.l2_norm_clip = l2_norm_clip
        self.secure_mode = secure_mode

    def compute_grad(self, x, y):
        error = self.predict(x) - y
        # 逐样本梯度
        batch_dw = np.expand_dims(x, axis=2) @ np.expand_dims(error, axis=1)
        # L2 裁剪
        batch_grad_l2 = np.sqrt((batch_dw**2).sum(axis=(1,2)))
        clip_factor = np.maximum(1.0, batch_grad_l2 / self.l2_norm_clip)
        batch_dw = batch_dw / clip_factor[:, np.newaxis, np.newaxis]
        dw = batch_dw.mean(axis=0) + self.alpha * self.weight
        # 加高斯噪声
        noise = np.random.normal(0, self.l2_norm_clip * self.noise_multiplier, dw.shape)
        dw = dw + noise
        return dw.squeeze()

    def compute_epsilon(self, steps, batch_size, delta):
        # RDP 会计: 计算 (ε, δ)-DP 隐私预算
        accountant = dp_accounting.rdp.RdpAccountant(orders)
        accountant.compose(SelfComposedDpEvent(
            PoissonSampledDpEvent(sampling_prob, GaussianDpEvent(noise)), steps))
        return accountant.get_epsilon(target_delta=delta)
```

---

## 3. HFL 逻辑回归

### 3.1 类层次

```
LogisticRegression (base.py)
  ├── Client (训练 + 通信)
  ├── LogisticRegression_DPSGD
  └── LogisticRegression_Paillier (泰勒近似)
```

### 3.2 模型实现

```python
class LogisticRegression:
    def __init__(self, x, y, learning_rate=0.2, alpha=0.0001):
        max_y = max(y)
        if max_y == 1:  # 二分类
            self.weight = np.zeros(x.shape[1])
            self.bias = np.zeros(1)
            self.multiclass = False
        else:           # 多分类 (One-vs-Rest)
            self.weight = np.zeros((x.shape[1], max_y + 1))
            self.bias = np.zeros((1, max_y + 1))
            self.multiclass = True

    def sigmoid(self, x):          # 数值稳定 Sigmoid
        pos_mask = x >= 0
        result = np.empty_like(x)
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        result[~pos_mask] = np.exp(x[~pos_mask]) / (np.exp(x[~pos_mask]) + 1)
        return result

    def predict_prob(self, x):
        return self.softmax(x) if self.multiclass else self.sigmoid(x)

    def compute_grad(self, x, y):
        error = self.predict_prob(x) - y
        dw = x.T.dot(error) / x.shape[0] + self.alpha * self.weight
        db = error.mean(axis=0, keepdims=True)
        return dw, db
```

### 3.3 Paillier 近似

Paillier 不支持密文乘法，因此用一阶泰勒展开近似 Sigmoid：

```python
# sigmoid(x) ≈ 0.5 + 0.25·x
# 这使得梯度下降在密文上只需加法 + 标量乘法
error = 0.5 + 0.25 * z - y     # z = X·θ
```

---

## 4. HFL 神经网络

### 4.1 MLP 结构

```
Input(D) → LazyLinear(32) → ReLU → LazyLinear(16) → ReLU → LazyLinear(K)
```

- 使用 PyTorch 构建
- `LazyLinear` 自动推断输入维度
- 支持分类（BCE/CE Loss）和回归（MSE Loss）
- DPSGD 通过 Opacus 库实现

### 4.2 CNN 结构

```
Conv2d(1→16, 8) → ReLU → AvgPool2d(2)
Conv2d(16→32, 4) → ReLU → AvgPool2d(2)
Linear(32*4*4 → 128) → ReLU → Linear(128 → K)
```

- 使用 torchvision 加载标准数据集
- 仅支持分类任务
- DPSGD 通过 Opacus 实现

---

## 5. VFL 线性/逻辑回归

### 5.1 类层次

```
VFL 三角色:
  Guest (特征方A)          Host (特征方B + 标签)    Coordinator
  LinearRegression_Guest_*  LinearRegression_Host_*  CKKSCoordinator
      ├── Plaintext             ├── Plaintext              │
      └── CKKS                  └── CKKS                   │
                                                           │
  LogisticRegressionGuest_*   LogisticRegressionHost_*      │
      ├── Plaintext             ├── Plaintext              │
      └── CKKS                  └── CKKS                   │
```

### 5.2 VFL 前向传播

```python
# Guest 端
class LinearRegression_Guest_Plaintext:
    def compute_z(self, x):
        return x.dot(self.weight)

# Host 端
class LinearRegression_Host_Plaintext:
    def compute_z(self, x, guest_z):
        z = x.dot(self.weight) + self.bias
        z += guest_z           # 合并双方计算结果
        return z

    def compute_error(self, y, z):
        return z - y

    def compute_grad(self, x, error):
        dw = x.T.dot(error) / x.shape[0] + self.alpha * self.weight
        db = error.mean(axis=0, keepdims=True)
        return dw, db
```

### 5.3 VFL CKKS 模式

```python
class CKKSCoordinator:
    def __init__(self):
        # 创建 TenSEAL CKKS 上下文
        self.context = ts.context(ts.SCHEME_TYPE.CKKS,
                                  poly_modulus_degree=8192,
                                  coeff_mod_bit_sizes=[60, 49, 49, 60])
        self.context.generate_galois_keys()
        self.context.global_scale = 2**49

    def decrypt_refresh(self, ct_host, ct_guest):
        """定期解密并重加密以控制噪声"""
        pt_host = self.context.decrypt(ct_host)
        pt_guest = self.context.decrypt(ct_guest)
        ct_host_new = self.context.encrypt(pt_host)
        ct_guest_new = self.context.encrypt(pt_guest)
        return ct_host_new, ct_guest_new
```

---

## 6. VFL XGBoost

### 6.1 架构

```
Guest (特征方)                    Host (标签方)
    │                                │
    │  PSI 样本对齐 ◄───────────────►│
    │                                │
    │  for tree in range(num_tree):  │
    │    │                           │
    │    │  E(g, h) ◄────────────── │  计算梯度、海森矩阵
    │    │                           │  g = ∂L/∂pred
    │    │  G = Σ E(g)               │  h = ∂²L/∂pred²
    │    │  H = Σ E(h)               │
    │    │                           │
    │    │  构建树 (分桶 + GOSS)     │
    │    │  gain = GL²/HL + GR²/HR   │
    │    │                           │
    │    │  pred ──────────────────► │  更新预测值
    │    │                           │
```

### 6.2 Paillier 梯度加密

```python
# 梯度合并: 将 g 和 h 编码为一个整数
# 放大 10^4 倍保持精度
g_int = int(g * 10**4)
h_int = int(h * 10**4)

# 合并为一个值 (可选)
merge_gh = g_int * 10**8 + h_int

# 并行加密 (Ray ActorPool)
encrypted = pai_actor.pai_enc.remote(merge_gh)

# 密文聚合
enc_sum = atom_paillier_sum(encrypted)

# 解密
decrypted = pai_actor.pai_dec.remote(enc_sum)
g_sum = decrypted // 10**8
h_sum = decrypted % 10**8

# 缩回原始尺度
g_agg = g_sum / 10**4
h_agg = h_sum / 10**4
```

### 6.3 GOSS 采样

```python
# GOSS (Gradient-based One-Side Sampling)
# 保留大梯度样本 + 随机采样小梯度样本
n_total = len(grad)
n_large = int(n_total * large_ratio)
n_small = int(n_total * small_ratio)

top_indices = np.argsort(np.abs(grad))[::-1][:n_large]
remaining = np.setdiff1d(arange(n), top_indices)
random_indices = np.random.choice(remaining, size=n_small)
sampled = np.concatenate([top_indices, random_indices])

# 小梯度样本权重放大
weight = ones(n_small) * n_total / n_small / large_ratio_reciprocal
```

---

## 7. HFL ChatGLM

### 7.1 架构

```
Server (聚合方)
    │
    ├──► Client 1: ChatGLM 本地微调 (P-Tuning v2)
    ├──► Client 2: ChatGLM 本地微调
    └──► Client N: ChatGLM 本地微调
    │
    仅聚合 prefix_encoder 权重 (占总参数量 < 0.1%)
```

### 7.2 P-Tuning v2

```python
# 只微调 prefix_encoder，冻结主模型
class ChatGlmClient:
    def __init__(self):
        self.model = AutoModel.from_pretrained("chatglm")
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 只训练 prefix_encoder
        self.prefix_encoder = PrefixEncoder(config)
        self.prefix_encoder.train()

    def get_trainable_weights(self):
        # 提取 prefix_encoder 参数用于聚合
        return [p.data.cpu().numpy() for p in self.prefix_encoder.parameters()]

    def set_weights(self, weights):
        for p, w in zip(self.prefix_encoder.parameters(), weights):
            p.data = torch.from_numpy(w).to(p.device)
```

---

## 8. 预处理流水线

### 8.1 HFL 预处理器

每个预处理器继承 sklearn 风格接口并添加 FL 通信：

```python
class StandardScaler(FL_BaseEstimator):
    def fit(self, X):
        local_mean = X.mean(axis=0)
        local_std = X.std(axis=0)
        n = X.shape[0]

        # FL 聚合 (HFL 模式)
        if self.FL_type == 'H':
            self.channel.send("mean", local_mean)
            self.channel.send("std", local_std)
            self.channel.send("n", n)
            # 接收全局聚合结果
            global_mean = self.channel.recv("global_mean")
            global_std = self.channel.recv("global_std")
            self.module.mean_ = global_mean
            self.module.scale_ = global_std
```

### 8.2 支持的预处理器（17 个）

| 类别 | 类名 | sklearn 对应 |
|------|------|-------------|
| 缩放 | `StandardScaler` | `sklearn.preprocessing.StandardScaler` |
| 缩放 | `MinMaxScaler` | `sklearn.preprocessing.MinMaxScaler` |
| 缩放 | `MaxAbsScaler` | `sklearn.preprocessing.MaxAbsScaler` |
| 缩放 | `RobustScaler` | `sklearn.preprocessing.RobustScaler` |
| 缩放 | `Normalizer` | `sklearn.preprocessing.Normalizer` |
| 编码 | `OneHotEncoder` | `sklearn.preprocessing.OneHotEncoder` |
| 编码 | `OrdinalEncoder` | `sklearn.preprocessing.OrdinalEncoder` |
| 编码 | `TargetEncoder` | `sklearn.preprocessing.TargetEncoder` |
| 编码 | `LabelEncoder` | `sklearn.preprocessing.LabelEncoder` |
| 编码 | `LabelBinarizer` | `sklearn.preprocessing.LabelBinarizer` |
| 编码 | `MultiLabelBinarizer` | `sklearn.preprocessing.MultiLabelBinarizer` |
| 填充 | `SimpleImputer` | `sklearn.impute.SimpleImputer` |
| 离散 | `KBinsDiscretizer` | `sklearn.preprocessing.KBinsDiscretizer` |
| 变换 | `PowerTransformer` | `sklearn.preprocessing.PowerTransformer` |
| 变换 | `QuantileTransformer` | `sklearn.preprocessing.QuantileTransformer` |
| 变换 | `SplineTransformer` | `sklearn.preprocessing.SplineTransformer` |
| 组合 | `Pipeline` | `sklearn.pipeline.Pipeline` |

---

## 9. FL 统计

### 9.1 统计函数

| 函数 | 计算内容 | HFL | VFL |
|------|---------|-----|-----|
| `col_mean` | 列均值 | ✓ | ✓ |
| `col_var` | 列方差 | ✓ | ✓ |
| `col_min` | 列最小值 | ✓ | ✓ |
| `col_max` | 列最大值 | ✓ | ✓ |
| `col_min_max` | 列最小最大值 | ✓ | ✓ |
| `row_min` | 行最小值 | ✗ | ✓ |
| `row_max` | 行最大值 | ✗ | ✓ |
| `col_sum` | 列求和 | ✓ | ✓ |
| `row_sum` | 行求和 | ✗ | ✓ |
| `col_norm` | 列 L2 范数 | ✓ | ✓ |
| `row_norm` | 行 L2 范数 | ✗ | ✓ |
| `col_frequent` | 众数统计 | ✓ | ✗ |
| `col_union` | 并集统计 | ✓ | ✗ |
| `col_quantile` | 分位数 (KLL sketch) | ✓ | ✗ |

### 9.2 假设检验

```python
class FederatedHypothesisTests:
    def ttest(self, X, columns):
        """联邦 T 检验"""
        data1 = X[:, columns[0]]
        data2 = X[:, columns[1]]
        # 本地统计量
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = data1.mean(), data2.mean()
        var1, var2 = data1.var(ddof=1), data2.var(ddof=1)
        # FL 聚合后计算 t_stat, p_value
        ...
```

---

## 10. 配置格式

### 10.1 任务配置 (JSON)

```json
{
  "party_info": {
    "task_manager": "127.0.0.1:50050"
  },
  "component_params": {
    "roles": {
      "host": "Bob",
      "guest": ["Charlie"]
    },
    "common_params": {
      "model": "HeteroXGB",
      "task_name": "train",
      "psi": "KKRT",
      "num_tree": 5,
      "max_depth": 5,
      "learning_rate": 0.1
    },
    "role_params": {
      "Bob": {
        "data_set": "train_host",
        "id": "id",
        "label": "y",
        "model_path": "data/result/host_model.pkl"
      },
      "Charlie": {
        "data_set": "train_guest",
        "model_path": "data/result/guest_model.pkl"
      }
    }
  }
}
```

### 10.2 HFL 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | string | Plaintext | Plaintext / DPSGD / Paillier |
| `process` | string | train | train / predict |
| `global_epoch` | int | 10 | 全局聚合轮数 |
| `local_epoch` | int | 1 | 本地训练轮数 |
| `batch_size` | int | 32 | 批处理大小 |
| `learning_rate` | float | 0.2 | 学习率 |
| `alpha` | float | 0.0001 | L2 正则化系数 |
| `id` | string | id | ID 列名 |
| `label` | string | y | 标签列名 |
| `noise_multiplier` | float | 1.0 | DPSGD 噪声乘数 |
| `l2_norm_clip` | float | 1.0 | DPSGD 梯度裁剪阈值 |
| `delta` | float | 1e-5 | DPSGD delta 参数 |

### 10.3 VFL 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `method` | string | Plaintext | Plaintext / CKKS |
| `encrypted_proto` | string | paillier | 加密协议 (XGBoost) |
| `psi` | string | KKRT | PSI 协议 |
| `secure_bits` | int | 112 | XGBoost 密钥位数 |
| `actors` | int | 20 | Ray Actor 数 |
| `merge_gh` | bool | true | 梯度合并 |

---

## 11. 数据流

### 11.1 HFL 数据流向

```
数据文件 (CSV)
    │
    ▼
read_data(data_info) → pandas.DataFrame
    │
    ▼
StandardScaler.fit_transform(x) → 标准化数据
    │
    ▼
DataLoader(x, y, batch_size) → 批次迭代器
    │
    ▼
Local Training: model.fit(batch_x, batch_y)  × local_epoch
    │
    ▼
grpc.send("client_model", theta) → Server 聚合
    │
    ▼
Server: weighted_average(client_models)
    │
    ▼
grpc.send_all("server_model", theta_avg) → Client 更新
    │
    ▼  × global_epoch
模型保存: save_pickle_file(model, path)
指标保存: save_json_file(metrics, path)
```

### 11.2 VFL 数据流向

```
三方独立读取 CSV → PSI 样本对齐 → 特征切分
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
             Guest 数据             Host 数据          Coordinator
                  │                    │
                  ▼                    ▼
           CKKS 加密模型           CKKS 加密模型     CKKS 密钥生成
                  │                    │
                  ▼                    ▼
           前向计算 z_guest        前向计算 z_host
                  │                    │
                  └────────┬───────────┘
                           ▼
                   计算 error
                           │
                    ┌──────┴──────┐
                    ▼              ▼
                Guest 梯度      Host 梯度
                    │              │
                    ▼              ▼
              本地模型更新      本地模型更新
```

---

## 12. 密码学封装

### 12.1 Paillier 封装

```python
class Paillier:
    def __init__(self, public_key, private_key):
        self.public_key = public_key
        self.private_key = private_key

    def encrypt_scalar(self, plain):       return self.public_key.encrypt(plain)
    def encrypt_vector(self, vec):         return [self.public_key.encrypt(i) for i in vec]
    def encrypt_matrix(self, mat):         return [[self.public_key.encrypt(i) for i in row] for row in mat]
    def decrypt_scalar(self, cipher):      return self.private_key.decrypt(cipher)
    def decrypt_vector(self, vec):         return [self.private_key.decrypt(i) for i in vec]
    def decrypt_matrix(self, mat):         return [[self.private_key.decrypt(i) for i in row] for row in mat]
```

### 12.2 CKKS 封装

```python
class CKKS:
    def __init__(self, context):
        self.context = context
        self.multiply_depth = context.data.seal_context().first_context_data().chain_index()

    def encrypt_vector(self, vec):
        return ts.ckks_vector(self.context, vec)
    def encrypt_tensor(self, tensor):
        return ts.ckks_tensor(self.context, tensor)
    def decrypt(self, ct):
        return ct.decrypt()
    def load_vector(self, vec_str):
        return ts.ckks_vector_from(self.context, vec_str)
```
