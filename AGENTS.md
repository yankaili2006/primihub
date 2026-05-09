# PrimiHub 项目分析与改进方案

## 项目功能

PrimiHub 是开源隐私计算平台，核心能力：

- **PSI** (隐私集合求交): ECDH-PSI, KKRT-PSI, TEE-PSI, OpenMined PSI, APSI
- **PIR** (隐私信息检索): ID-PIR, Keyword-PIR
- **联邦学习 FL**: 横向/纵向 LR, 神经网络, CNN, XGBoost, ChatGLM
- **MPC** (安全多方计算): ABY3 秘密共享, CrypTFlow2, Falcon 安全 CNN
- **同态加密**: Paillier, CKKS, SEAL
- **TEE**: Intel SGX 远程证明, DCAP
- **数据接入**: CSV/SQLite/MySQL/HDFS/Parquet/Image

架构: 3 节点集群 + Java 元数据服务, C++ 核心 + Python SDK, Bazel 构建, gRPC/Apache Arrow Flight 通信。

---

## 改进方案

### P0 - 严重/阻塞

| # | 问题 | 位置 | 方案 |
|---|------|------|------|
| 1 | 硬编码 API Key `5oC06czJLeF3kXdfg7D1q2z0G4wwYJ3l` | `src/primihub/node/node_impl.cc:189` | 移到配置文件/环境变量 |
| 2 | TLS 验证关闭 (MITM 风险) | `node_impl.cc:203-204` | 启用 CURLOPT_SSL_VERIFYPEER |
| 3 | FL 通信 pickle.loads() 远程代码执行 | `python/primihub/FL/utils/net_work.py:56` | 替换为 JSON/protobuf/msgpack |
| 4 | KKRT PRNG 用 time(nullptr) | `kernel/psi/operator/kkrt_psi.cc:85` | 用 `oc::sysRandomSeed()` |
| 5 | TEE PSI 解密失败返回成功 | `kernel/psi/operator/tee_psi.cc:94-96` | 修复返回值 |
| 6 | OSS 镜像 403 导致编译失败 | WORKSPACE | 已切到 WORKSPACE_GITHUB, 覆盖 OSS-only deps |
| 7 | WORKSPACE 文件 7 份重复 | 项目根目录 | 只保留 WORKSPACE_GITHUB, 清理其他副本 |

### P1 - 高优先级

| # | 问题 | 位置 | 方案 |
|---|------|------|------|
| 8 | CSV/MySQL read(offset,limit) 返回 nullptr | data_store/ | 实现分页读取 |
| 9 | MySQL 驱动 SQL 注入 | `mysql/mysql_driver.cc:414-456` | 用参数化查询 |
| 10 | MySQL 无连接池 | `mysql/mysql_driver.cc:175-203` | 加连接池 |
| 11 | gRPC 重试无退避 | `util/network/grpc_link_context.h:72` | 指数退避 |
| 12 | 8 个算法 stub 未实现 | `task/semantic/mpc_task.cc:39-56` | 实现或删除 |
| 13 | 已知加密 bug | `algorithm/opt_paillier/src/utils.cc:28` | 修复 ceil_log2 |
| 14 | no gRPC 认证/授权 | `node/node_interface.cc` | 加 middleware |
| 15 | 模型默认 PUBLIC 可见性 | `algorithm/logistic.cc:553` | 默认 PRIVATE |
| 16 | 调试日志输出 MySQL 密码 | `data_store/mysql/mysql_driver.cc:92` | 移除 |

### P2 - 性能优化

| # | 问题 | 位置 | 方案 |
|---|------|------|------|
| 17 | 任务调度用裸 std::thread | 多处 scheduler | 用线程池 |
| 18 | FL 服务端全量加载模型 | `FL/neural_network/hfl_server.py:150-164` | 流式聚合 |
| 19 | 客户端 3MB 分片多余 | `common.h:23` vs server 128MB | 移除或调整分片大小 |
| 20 | LR 数据加载重复扫描 Arrow chunks | `algorithm/logistic.cc:235-298` | 批量读取 |
| 21 | 进程参数泄漏任务数据 | `node/worker/worker.cc:169-172` | 用文件传递 |

### P3 - 代码质量/可维护性

| # | 问题 | 位置 | 方案 |
|---|------|------|------|
| 22 | Proto 拼写 DownloadRespone | `protos/service.proto:11` | 修正 |
| 23 | fl_task_test.cc 断言全注释掉 | `test/primihub/task/fl_task_test.cc:15` | 恢复断言 |
| 24 | express_test.cc 断言禁用 | `test/primihub/executor/express_test.cc:80` | 恢复 |
| 25 | 5/6 测试源文件被注释 | `test/primihub/protocol/BUILD` | 启用 |
| 26 | Python 全局可变单例 | `python/primihub/context.py:66` | 改为依赖注入 |
| 27 | Makefile format 只格式化 .h | `Makefile:183` | 修复 find 通配符 |
| 28 | Python 生成 protobuf 提交到 git | `ph_grpc/src/primihub/protos/` | .gitignore + 构建时生成 |
| 29 | Docker Compose YAML 重复 60% | `docker-compose.yml` | 用 YAML anchors |
| 30 | __init__.py 暴露 TODO | `python/primihub/__init__.py:11` | 修复 |
| 31 | executor.py 硬编码路径 | `python/primihub/executor.py:54` | 用包相对路径 |

---

## 外部依赖来源

编译过程依赖以下第三方资源：

### GitHub (主要来源, ~120+ 个 URL)
- **github.com/primihub/** (21 个私有 fork): bazel-rules-thirdparty, grpc, arrow, libPSI, libOTe, aby3, cryptoTools, PSI, PIR, APSI, SEAL, ntl, curl, served, TEE, communication, rules_boost, cpp-base64, soralog, relic, upb
- **github.com/bazelbuild/** (20+): rules_foreign_cc, platforms, bazel-skylib, rules_cc, rules_java, rules_proto, rules_python, rules_pkg, rules_go, bazel-gazelle, rules_apple, apple_support, rules_rust, bazel-toolchains
- **github.com/google/** (12+): glog, flatbuffers, gflags, googletest, benchmark, boringssl, sparsehash, re2, leveldb, libprotobuf-mutator, snappy, brotli, double-conversion, abseil-cpp
- **github.com/protocolbuffers/** (2): protobuf, upb
- **github.com/** 其他 (30+): nlohmann/json, pybind/pybind11, openssl, facebook/zstd, apache/thrift, lz4, redis/hiredis, microsoft/GSL/Kuku, pocoproject/poco, jbeder/yaml-cpp, fmtlib/fmt, boost-ext/di, jupp0r/prometheus-cpp, grpc/grpc, open-telemetry/opentelemetry-cpp, etc.

### Alibaba Cloud OSS (已被废弃, 403)
- `primihub.oss-cn-beijing.aliyuncs.com` (~65 个文件) - 原始 CN 镜像
- 大部分文件也有 GitHub 来源, 但 `sgxsdk.tar.gz` 和部分自定义工具仅在此 OSS 上有

### Google Cloud Storage (gRPC 依赖镜像)
- `storage.googleapis.com/grpc-bazel-mirror` (~25 个文件) - gRPC 生态的备份源

### Bazel 官方镜像
- `mirror.bazel.build` (~15 个 URL) - Bazel 核心包

### 其他域名 (7 个)
- `sourceware.org` (bzip2), `gmplib.org` (GMP), `zlib.net` (zlib)
- `gitlab.com/libeigen` (Eigen), `fossies.org` (bzip2 fallback)
- `gitlab.primihub.com` (private-join-and-compute)
- `gitee.com/primihub` (cityhash - 仅 CN 版本使用)

### Python 包索引
- `files.pythonhosted.org` / `pypi.python.org` (gRPC Python deps)

---

## 构建过程要点

1. **Bazel 5.0.0**: 通过 `.bazelversion` 锁定, 使用 `bazel` 命令
2. **WORKSPACE**: 当前使用 `WORKSPACE_GITHUB` (已替换原始 WORKSPACE)
3. **核心二进制**: `//:node` (primihub-node), `//:cli` (primihub-cli), `//:task_main`
4. **构建命令**: `make release` 或 `bazel build --config=linux_x86_64 //:node //:cli //:task_main`
5. **Python 绑定**: 通过 `disable_py_task=y` 禁用 (避免 pybind11/Python 版本兼容问题)
6. **MySQL**: 通过 `mysql=` 禁用 (无 libmysqlclient)
7. **本地依赖**: `local_deps/` 目录下的 zlib/private-join-and-compute 归档
