# PrimiHub MPC 算子性能测试方案

## 1. 测试目标

| 目标 | 说明 |
|------|------|
| 基准性能 | 获取各 MPC 算子在不同数据规模下的执行耗时和吞吐量 |
| 对比分析 | 对比不同算子类型（本地运算 vs 安全交互）的性能差异 |
| 扩展性评估 | 评估算子性能随数据规模增长的扩展性 |
| 网络开销 | 分析网络往返（RTT）对算子延迟的影响 |

---

## 2. 测试环境

### 2.1 硬件环境

| 配置项 | 规格 |
|--------|------|
| CPU | 至少 4 核，记录 CPU 型号和频率 |
| 内存 | 至少 16 GB |
| 网络 | 本地回环测试（127.0.0.1）或跨机器 |
| 操作系统 | Linux x86_64 / aarch64 |

### 2.2 软件环境

| 组件 | 版本 |
|------|------|
| Bazel | 5.0+ |
| C++ 标准 | C++17 |
| 编译器 | GCC 9+ / Clang 12+ |
| ABY3 | PrimiHub fork |

---

## 3. 测试指标

| 指标 | 单位 | 说明 |
|------|------|------|
| 平均耗时 | ms | 多次运行的平均执行时间 |
| 最小耗时 | ms | 多次运行中的最小值 |
| 最大耗时 | ms | 多次运行中的最大值 |
| 吞吐量 | ops/s | 每秒处理的元素数 = 元素数 / 平均耗时(秒) |
| 标准差 | ms | 结果的离散程度 |
| 网络流量 | bytes | 单次操作的总通信量 |

---

## 4. 测试算子范围

### 4.1 基础算术算子

| 算子 | 方法 | 通信轮数 | 计算复杂度 | 安全性 |
|------|------|---------|-----------|--------|
| ADD | `MPC_Add` | 0 | O(n) | 仅本地运算 |
| ADD_CONST | `MPC_Add_Const` | 0 | O(n) | 仅本地运算 |
| SUB | `MPC_Sub` | 0 | O(n) | 仅本地运算 |
| SUB_CONST | `MPC_Sub_Const` | 0 | O(n) | 仅本地运算 |
| MUL | `MPC_Mul` | 1 | O(n) | 需 Beaver Triple |
| MUL_CONST | `MPC_Mul_Const` | 0 | O(n) | 仅本地运算 |
| DOT_MUL | `MPC_Dot_Mul` | 1 | O(n) | 需 Beaver Triple |
| DIV | `MPC_Div` | ~8 | O(n·iter) | Goldschmidt 迭代 |
| CMP | `MPC_Compare` | 1 | O(n·logW) | MSB 电路 |

### 4.2 高级统计算子

| 算子 | 实现类 | 交互次数 |
|------|--------|---------|
| SUM | `MPCSumOrAvg` | 0（本地聚合后 reveal） |
| AVG | `MPCSumOrAvg` | 0（SUM + 本地除法） |
| MAX | `MPCMinOrMax` | logN 次 CMP |
| MIN | `MPCMinOrMax` | logN 次 CMP |
| T_TEST | `MPCTTest` | 2（share + reveal） |
| F_TEST | `MPCFTest` | 2（share + reveal） |
| CHI_SQUARE | `MPCChiSquareTest` | 2（share + reveal） |
| REGRESSION | `MPCRegression` | 2（share + reveal） |
| CORRELATION | `MPCCorrelation` | 2（share + reveal） |

---

## 5. 测试数据规模

### 5.1 数据量级

| 级别 | 元素数 | 说明 |
|------|--------|------|
| S（小） | 64 | 单次小规模计算 |
| M（中） | 256 | 常规计算规模 |
| L（大） | 1,024 | 批量计算 |
| XL（超大） | 10,000+ | 大规模计算（可选） |

### 5.2 数据生成策略

```cpp
// 整数类型 (si64)
eMatrix<i64> input(rows, cols);
for (u64 i = 0; i < rows; i++)
  for (u64 j = 0; j < cols; j++)
    input(i, j) = static_cast<i64>(i + j + 1);  // 递增序列

// 浮点类型 (sf64)
eMatrix<double> input(rows, cols);
for (u64 i = 0; i < rows; i++)
  input(i, 0) = (i % 2 == 0) ? (i + 1.0) : -(i + 1.0);  // 正负数混合
```

---

## 6. 测试方法

### 6.1 测量框架

```
每个测试用例执行流程：
  1. Warmup:    执行 1 次（预热缓存和连接）
  2. Iteration: 执行 N 次（默认 5 次）
  3. 收集耗时:  high_resolution_clock
  4. 统计:      avg / min / max / stddev / throughput
```

### 6.2 统计公式

```
avg_time  = Σ(t_i) / N
min_time  = min(t_i)
max_time  = max(t_i)
stddev    = sqrt(Σ(t_i - avg)² / (N-1))
throughput = data_size / (avg_time / 1000.0)   // ops/s
```

### 6.3 多轮迭代

为避免单次运行的随机误差，每个测试用例：
- 预热 1 轮（丢弃结果）
- 执行 5 轮取平均值

对于 DIV/CMP 等重算子，预热 0 轮，执行 3 轮。

---

## 7. 测试分类

### 7.1 单算子基准测试

测试每个算子在给定数据规模下的执行耗时。

**预期输出格式**：

```
==============================================
  PrimiHub MPC Operator Performance Benchmark
==============================================

Operator    Elements    Avg (ms)        Throughput
--------------------------------------------------
ADD         64          0.012           5333333.333 ops/s
ADD         256         0.045           5688888.889 ops/s
ADD         1024        0.178           5752808.989 ops/s
MUL         64          0.089           719101.124 ops/s
MUL         256         0.352           727272.727 ops/s
MUL         1024        1.410           726241.135 ops/s
DIV         64          5.234           12228.506 ops/s
DIV         256         20.891          12254.032 ops/s
CMP         64          0.456           140350.877 ops/s
CMP         256         1.823           140427.866 ops/s
...
```

### 7.2 数据规模扩展性测试

对每个算子测试从小到大的数据量级，评估时间复杂度：

| 算子 | 理论复杂度 | 验证方法 |
|------|-----------|---------|
| ADD/SUB | O(n) | 耗时与 n 呈线性关系 |
| MUL/DOT_MUL | O(n) | 耗时与 n 呈线性关系 |
| CMP | O(n·logW) | n 翻倍时耗时接近线性增长 |
| DIV | O(n·iter) | n 翻倍时耗时接近线性增长（iter ≈ 2） |

### 7.3 网络延迟影响测试

模拟不同网络延迟（通过 `tc` 工具注入延迟）：

```bash
# 添加 10ms 延迟
tc qdisc add dev lo root netem delay 10ms

# 运行测试
bazel run //test/primihub/benchmark:bench_mpc

# 清理
tc qdisc del dev lo root
```

| 延迟 | 说明 |
|------|------|
| 0ms | 本地回环（基准） |
| 1ms | 同机房内网 |
| 10ms | 同城跨机房 |
| 50ms | 跨地域网络 |

### 7.4 并发扩展性测试

测试不同并发 Worker 数量下的吞吐量：

| Worker 数 | 说明 |
|-----------|------|
| 1 | 单线程基准 |
| 2 | 双 Worker 并发 |
| 4 | 四 Worker 并发 |

---

## 8. 测试执行

### 8.1 命令行执行

```bash
# 构建基准测试
bazel build //test/primihub/benchmark:bench_mpc

# 运行（自动 fork 3 个子进程）
./bazel-bin/test/primihub/benchmark/bench_mpc

# 或者通过 bazel test
bazel test --test_output=all //test/primihub/benchmark:bench_mpc --test_timeout=300
```

### 8.2 输出到 CSV

```bash
# 运行并捕获输出
./bazel-bin/test/primihub/benchmark/bench_mpc 2>/dev/null | tee benchmark_results.txt
```

---

## 9. 结果分析模板

### 9.1 性能基线记录

```
测试日期: YYYY-MM-DD
测试环境: CPU / RAM / OS / 编译器
网络配置: 本地回环 (latency=0ms)

=== 算子性能基线 ===
| Operator | 64 els (ms) | 256 els (ms) | 1024 els (ms) | 吞吐量 (ops/s) |
|----------|------------|-------------|--------------|----------------|
| ADD      |            |             |              |                |
| SUB      |            |             |              |                |
| MUL      |            |             |              |                |
| DOT_MUL  |            |             |              |                |
| DIV      |            |             |              |                |
| CMP      |            |             |              |                |
| ABS      |            |             |              |                |
| DReLu    |            |             |              |                |
```

### 9.2 性能对比

```
=== 本地运算 vs 安全交互对比 ===
本地运算 (ADD/SUB):         ~0.01ms (64 elements)
需交互运算 (MUL/DOT_MUL):    ~0.09ms (64 elements)
重交互运算 (DIV):            ~5.2ms  (64 elements)
```

### 9.3 异常检测

运行测试过程中应观察：
- 是否有算子执行时间异常偏高（可能为网络问题）
- 是否有算子执行失败或产生错误结果
- 吞吐量是否与数据规模呈预期关系

---

## 10. 自动化集成

### 10.1 CI 集成

在 `.github/workflows/main.yml` 中已配置：

```yaml
- name: benchmark test
  run: |
    bazel build //test/primihub/benchmark:bench_mpc
    ./bazel-bin/test/primihub/benchmark/bench_mpc
```

### 10.2 回归检测

基准测试结果应纳入版本管理，每次代码变更后运行以检测性能回归：

```bash
# 保存基准
./bazel-bin/test/primihub/benchmark/bench_mpc > baseline.txt

# 修改后重新测试
./bazel-bin/test/primihub/benchmark/bench_mpc > current.txt

# 对比
diff baseline.txt current.txt
```

---

## 11. 附录

### 11.1 测试代码结构

```
test/primihub/benchmark/
├── BUILD              # Bazel 构建配置
└── bench_mpc.cc       # 基准测试主程序 (465 行)

test/primihub/protocol/
└── BUILD              # 协议层测试（含单元测试）
```

### 11.3 基准测试实现

`bench_mpc.cc` 实现了 **15 个 MPC 算子的性能测试**，覆盖 3 种数据规模。

**数据准备**:
- `make_shares()` — 创建 si64 类型秘密分享
- `make_fp_shares()` — 创建 sf64 (定点数) 类型秘密分享

**测量方法**:
```cpp
// 1. 采集原始耗时数据
auto times = collect_times(f, warmup=2, iterations=7);

// 2. 统计分析
auto result = analyze(name, dtype, rows, cols, times, category);
// 输出: avg_ms, stddev_ms, min_ms, max_ms, throughput
```

**输出格式**:
- 控制台表格（含 Operator / Type / Rows / Cols / Elements / Avg / StdDev / Min / Max / Throughput / Category）
- CSV 文件 `benchmark_results.csv`（供图表分析）
- 按 Category 汇总（local / interactive / circuit / piecewise / heavy）

**算子分类体系**:

| Category | 含义 | 算子 |
|----------|------|------|
| `local_nocomm` | 本地运算，无通信 | ADD, SUB, ADD_CONST, SUB_CONST, MUL_CONST |
| `interactive` | 需 1 轮 Beaver Triple 交互 | MUL, DOT_MUL |
| `interactive_circuit` | 布尔电路 | CMP (MSB) |
| `interactive_piecewise` | 分段函数 | ABS, DReLu, Quo |
| `interactive_heavy` | 多轮迭代 | DIV, Pow, Pow2 |

**运行方式**:
```bash
bazel build //test/primihub/benchmark:bench_mpc
./bazel-bin/test/primihub/benchmark/bench_mpc
```

### 11.2 依赖项

| Bazel 目标 | 用途 |
|-----------|------|
| `//src/primihub/operator:aby3_operator` | ABY3 算子库 |
| `//src/primihub/protocol:protocol_aby3_lib` | ABY3 协议库 |
| `//src/primihub/util/network:network_lib` | 网络通信 |
| `@com_google_googletest//:gtest_main` | 测试框架 |
| `@com_github_grpc_grpc//:grpc++` | gRPC 通信 |
