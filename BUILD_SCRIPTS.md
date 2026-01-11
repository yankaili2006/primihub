# Docker 镜像构建脚本说明

本目录包含三个 Docker 镜像构建脚本，适用于不同的使用场景。

## 快速开始

### 推荐构建流程 (2026-01-09)

如果遇到基础镜像拉取问题，推荐使用以下增量构建方式：

```bash
# 1. 登录阿里云镜像仓库
echo "your_password" | docker login --username=primihub --password-stdin registry.cn-beijing.aliyuncs.com

# 2. 运行预构建脚本
bash pre_build.sh

# 3. 编译项目
make release mysql=y

# 4. 使用 Dockerfile.build 构建镜像
docker build -t primihub/primihub-node:1.8.0 -f Dockerfile.build .
```

**优势**:
- ✅ 基于现有镜像增量更新，速度快
- ✅ 避免基础镜像拉取问题
- ✅ 减少网络依赖
- ✅ 构建时间短（约 1-2 分钟）

### 标准构建流程

```bash
# 使用 build_docker.sh 完整构建
bash build_docker.sh FULL 1.8.0 primihub/primihub-node
```

## 脚本列表

| 脚本 | 用途 | 推荐场景 |
|------|------|---------|
| `build_local.sh` | 基础本地构建脚本 | 本地开发和测试 |
| `build_docker.sh` | 完整构建脚本 | 手动构建，支持 FULL/MINI 模式 |
| `jenkins_build.sh` | CI/CD 构建脚本 | Jenkins 集成，支持推送到镜像仓库 |

---

## 1. build_local.sh - 基础本地构建

**用途**: 本地快速构建 Docker 镜像

### 使用方法

```bash
# 新格式 (推荐)
bash build_local.sh [MODE] [TAG] [IMAGE_NAME]

# 旧格式 (兼容)
bash build_local.sh [TAG] [IMAGE_NAME]
```

### 参数说明

- `MODE`: 编译模式 (FULL/MINI)，可选，默认 FULL
- `TAG`: 镜像标签，默认当前时间戳
- `IMAGE_NAME`: 镜像名称，默认 `primihub/primihub-node`

### 示例

```bash
# 使用默认参数构建 FULL 模式
bash build_local.sh

# 指定 MINI 模式
bash build_local.sh MINI

# 完整参数
bash build_local.sh FULL 2024-01-08 primihub/primihub-node

# 兼容旧版本调用
bash build_local.sh v1.0.0 primihub/primihub-node
```

### 输出

- **FULL 模式**: `${IMAGE_NAME}:${TAG}`
- **MINI 模式**: `${IMAGE_NAME}:mini-${TAG}`

---

## 2. build_docker.sh - 完整构建脚本

**用途**: 提供更详细的构建过程和日志输出

### 使用方法

```bash
bash build_docker.sh [MODE] [TAG] [IMAGE_NAME]
```

### 参数说明

- `MODE`: 编译模式 (FULL/MINI)，默认 FULL
- `TAG`: 镜像标签，默认当前时间戳
- `IMAGE_NAME`: 镜像名称，默认 `primihub/primihub-node`

### 特性

- ✅ 详细的步骤输出
- ✅ 版本信息记录 (git branch/commit)
- ✅ 自动清理临时文件
- ✅ 构建失败时中止
- ✅ 提供推送和运行命令提示

### 示例

```bash
# 使用默认参数
bash build_docker.sh

# 构建 MINI 版本
bash build_docker.sh MINI latest

# 推送到私有仓库
bash build_docker.sh FULL 2024-01-08 192.168.99.10/primihub/primihub-node
```

### 构建流程

1. **预构建**: 运行 `pre_build.sh`
2. **编译**: 使用 Bazel 编译项目 (`make mysql=y`)
3. **记录版本**: 保存 git 分支和 commit 信息
4. **打包**: 创建 `bazel-bin.tar.gz`
5. **构建镜像**: 使用 `Dockerfile.local` 构建

---

## 3. jenkins_build.sh - CI/CD 构建脚本

**用途**: Jenkins 持续集成/部署，支持自动推送镜像

### 使用方法

```bash
bash jenkins_build.sh [COMPILE_MODE] [REGISTRY]
```

### 参数说明

- `COMPILE_MODE`: 编译模式 (ALL/MINI)，默认 ALL
- `REGISTRY`: Docker 仓库地址，默认 `192.168.99.10`

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|-------|
| `BUILD_TIMESTAMP` | 构建时间戳 | 自动生成 |
| `PUSH_IMAGE` | 是否推送镜像 (yes/no) | yes |
| `ALIYUN_PUSH` | 是否推送到阿里云 (yes/no) | no |

### 特性

- ✅ 支持多镜像仓库推送
- ✅ 阿里云镜像仓库支持
- ✅ 彩色日志输出
- ✅ 环境变量配置
- ✅ 自动清理临时文件
- ✅ 完整的错误处理

### 示例

#### 基础使用

```bash
# 默认构建并推送到私有仓库
bash jenkins_build.sh

# 构建 MINI 版本
bash jenkins_build.sh MINI

# 指定私有仓库
bash jenkins_build.sh ALL 192.168.99.10
```

#### 环境变量控制

```bash
# 只构建不推送
PUSH_IMAGE=no bash jenkins_build.sh ALL

# 推送到阿里云
ALIYUN_PUSH=yes bash jenkins_build.sh ALL

# 指定构建时间戳
BUILD_TIMESTAMP=20240108-120000 bash jenkins_build.sh ALL
```

#### Jenkins Pipeline 集成

```groovy
stage('Build') {
    steps {
        script {
            sh '''
                export BUILD_TIMESTAMP=${BUILD_TIMESTAMP}
                export PUSH_IMAGE=yes
                export ALIYUN_PUSH=yes

                if [ "$compile_mode" = "ALL" ]; then
                    bash jenkins_build.sh ALL 192.168.99.10
                else
                    bash jenkins_build.sh MINI 192.168.99.10
                fi
            '''
        }
    }
}
```

### 镜像推送

#### 私有仓库

默认推送到: `${REGISTRY}/primihub/primihub-node:${TAG}`

```bash
# 推送到私有仓库
bash jenkins_build.sh ALL 192.168.99.10
```

#### 阿里云镜像仓库

推送到: `registry.cn-beijing.aliyuncs.com/primihub/primihub-node:${BUILD_TIMESTAMP}`

```bash
# 同时推送到私有仓库和阿里云
ALIYUN_PUSH=yes bash jenkins_build.sh ALL
```

---

## 编译模式对比

| 模式 | build_local.sh | build_docker.sh | jenkins_build.sh |
|------|----------------|-----------------|------------------|
| 完整版 | FULL | FULL | ALL |
| 精简版 | MINI | MINI | MINI |

### FULL/ALL 模式

- 删除 `python/requirements.txt` 第一行
- 标签: `${TAG}`
- 适用场景: 生产环境

### MINI 模式

- 保留完整 requirements.txt
- 标签: `mini-${TAG}`
- 适用场景: 测试环境、资源受限环境

---

## 依赖要求

### 系统依赖

- Docker
- Git
- Python 3.8+
- Bazel (通过 `pre_build.sh` 安装)

### 构建依赖

```bash
# 运行预构建脚本
bash pre_build.sh

# 编译项目
make mysql=y
```

---

## 常见问题

### Q1: 构建失败怎么办？

**A**: 检查以下内容：

1. 确认 Docker 服务运行正常
2. 检查网络连接（需要下载依赖）
3. 确认 Python 版本为 3.8
4. 查看构建日志定位错误

```bash
# 检查 Docker
docker --version
docker ps

# 检查 Python
python3 --version
```

### Q1.1: Docker 无法拉取基础镜像 (primihub/primihub-base)

**问题描述**:
```
ERROR: failed to resolve source metadata for docker.io/primihub/primihub-base:latest:
failed to do request: Head "https://registry-1.docker.io/v2/primihub/primihub-base/manifests/latest":
dial tcp 108.160.170.43:443: i/o timeout
```

**原因**:
- Docker Hub 网络连接超时
- `primihub/primihub-base` 镜像不存在或无法访问
- Docker 代理配置未生效

**解决方案**:

**方案 1: 使用现有版本镜像作为基础** (推荐)

创建 `Dockerfile.build` 使用已有的镜像版本作为基础：

```dockerfile
FROM registry.cn-beijing.aliyuncs.com/primihub/primihub-node:1.7.0

WORKDIR /app

# 移除旧的构建产物
RUN rm -rf bazel-bin cli node task_main primihub-cli primihub-node \
    bazel-bin/cli bazel-bin/node bazel-bin/task_main \
    bazel-bin/src/primihub/pybind_warpper/*.so \
    bazel-bin/src/primihub/task/pybind_wrapper/*.so \
    python/primihub/FL/model/*.so \
    commit.txt 2>/dev/null || true

# 复制新的构建产物
ADD bazel-bin.tar.gz ./
COPY src/primihub/protos/ src/primihub/protos/
COPY commit.txt ./

# 重新安装 Python 包
RUN cd python && python3 setup.py develop

WORKDIR /app
EXPOSE 50050
```

构建命令：
```bash
# 先登录阿里云镜像仓库
echo "your_password" | docker login --username=primihub --password-stdin registry.cn-beijing.aliyuncs.com

# 使用新的 Dockerfile 构建
docker build -t primihub/primihub-node:1.8.0 -f Dockerfile.build .
```

**方案 2: 配置 Docker 镜像加速器**

编辑 `/etc/docker/daemon.json`:
```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1panel.live",
    "https://hub.rat.dev"
  ]
}
```

重启 Docker 服务：
```bash
sudo systemctl restart docker
```

**方案 3: 使用完整的 Dockerfile 从 ubuntu:20.04 构建**

如果有本地 ubuntu:20.04 镜像或能访问 Docker Hub，可以使用 `Dockerfile` 进行完整构建。

### Q2: 如何修改 pip 镜像源？

**A**: 编辑 `Dockerfile.local`，修改 pip install 命令：

```dockerfile
RUN python3 -m pip install --upgrade pip -i http://192.168.99.32/simple/ --trusted-host 192.168.99.32 \
  && pip install --no-cache-dir -r requirements.txt -i http://192.168.99.32/simple/ --trusted-host 192.168.99.32
```

### Q3: 镜像推送失败？

**A**: 检查以下内容：

1. 确认 Docker 已登录目标仓库

```bash
# 登录私有仓库
docker login 192.168.99.10

# 登录阿里云
docker login registry.cn-beijing.aliyuncs.com
```

2. 确认网络连接
3. 检查仓库权限

### Q4: 如何清理构建缓存？

**A**: 使用以下命令：

```bash
# 清理 Bazel 缓存
bazel clean --expunge

# 清理 Docker 缓存
docker system prune -a

# 删除构建产物
rm -f bazel-bin.tar.gz commit.txt
```

### Q5: 构建成功但 Docker 镜像构建失败

**问题描述**:
Bazel 编译成功，生成了 `bazel-bin.tar.gz`，但 Docker 镜像构建时失败。

**常见原因**:
1. 基础镜像无法拉取（见 Q1.1）
2. 网络问题导致 apt/pip 安装失败
3. Dockerfile 中的路径不正确

**解决方案**:

1. **检查构建产物是否完整**:
```bash
# 检查 tar 包是否存在且大小合理
ls -lh bazel-bin.tar.gz
# 应该在 70-80MB 左右

# 检查版本信息文件
cat commit.txt
```

2. **使用增量构建方式**:
如果已有旧版本镜像，使用 `Dockerfile.build` 进行增量更新而不是完全重建。

3. **检查 Docker 构建日志**:
```bash
# 使用 --progress=plain 查看详细日志
docker build --progress=plain -t primihub/primihub-node:1.8.0 -f Dockerfile.build .
```

### Q6: 阿里云镜像仓库登录失败

**问题描述**:
```
Error response from daemon: pull access denied for registry.cn-beijing.aliyuncs.com/primihub/primihub-node
```

**解决方案**:

```bash
# 使用正确的凭据登录
docker login registry.cn-beijing.aliyuncs.com
# 用户名: primihub
# 密码: [联系管理员获取]

# 或使用命令行方式
echo "your_password" | docker login --username=primihub --password-stdin registry.cn-beijing.aliyuncs.com
```

### Q7: 构建时间过长

**问题描述**:
Bazel 编译时间超过 10 分钟。

**优化建议**:

1. **使用多线程编译**:
```bash
# 使用 4 个线程
make release mysql=y jobs=4
```

2. **保留 Bazel 缓存**:
```bash
# 不要频繁执行 bazel clean
# 只在必要时清理缓存
```

3. **使用本地缓存的依赖**:
确保 `pre_build.sh` 已正确配置代理和依赖。

---

## 目录结构

```
primihub/
├── build_local.sh          # 基础构建脚本
├── build_docker.sh         # 完整构建脚本 (新增)
├── jenkins_build.sh        # CI/CD 构建脚本 (新增)
├── pre_build.sh            # 预构建脚本
├── Dockerfile              # 标准 Dockerfile
├── Dockerfile.local        # 本地构建 Dockerfile
├── Dockerfile.release      # 发布 Dockerfile
├── docker-compose.yml      # 容器编排配置
└── BUILD_SCRIPTS.md        # 本文档
```

---

## 更新日志

### 2026-01-09

- ✅ 新增 `Dockerfile.build` - 基于现有镜像的增量构建方案
- ✅ 添加 Docker 镜像构建故障排查指南
- ✅ 完善阿里云镜像仓库使用说明
- ✅ 添加网络问题解决方案
- ✅ 成功构建 primihub-node:1.8.0 镜像
- ✅ 优化构建流程文档

### 2024-01-08

- ✅ 新增 `build_docker.sh` - 完整构建脚本
- ✅ 新增 `jenkins_build.sh` - CI/CD 构建脚本
- ✅ 更新 `build_local.sh` - 支持 FULL/MINI 模式
- ✅ 添加阿里云镜像仓库支持
- ✅ 改进错误处理和日志输出
- ✅ 添加环境变量配置支持

---

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

Apache 2.0 License
