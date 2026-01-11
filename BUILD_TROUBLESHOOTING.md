# PrimiHub 构建问题解决指南

## 概述

本文档记录了在构建和运行PrimiHub过程中遇到的常见问题及其解决方案。基于实际测试经验整理。

## 问题分类

### 1. 编译依赖问题

#### 问题1: GMP库构建失败 - 缺少m4工具
**错误信息**:
```
configure: error: No usable m4 in $PATH or /usr/5bin (see config.log for reasons).
make: *** No targets specified and no makefile found.  Stop.
```

**原因**: GMP库构建需要m4工具进行宏处理。

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install m4

# RHEL/CentOS
sudo yum install m4

# macOS
brew install m4
```

#### 问题2: MySQL驱动编译失败
**错误信息**:
```
fatal error: mysql/mysql.h: No such file or directory
#include <mysql/mysql.h>
```

**原因**: 缺少MySQL开发库。

**解决方案**:
```bash
# Ubuntu/Debian
sudo apt-get install libmysqlclient-dev

# RHEL/CentOS
sudo yum install mysql-devel

# macOS
brew install mysql-client
```

**临时方案**: 如果不需MySQL支持，可以禁用MySQL驱动:
```bash
make release mysql=
```

### 2. 网络下载问题

#### 问题3: 依赖下载超时
**错误信息**:
```
failed to do request: Head "https://registry-1.docker.io/v2/...": dial tcp ...:443: i/o timeout
```

**原因**: 网络连接问题，特别是从国内访问国外资源。

**解决方案**:

**方案A: 使用HTTP代理**
```bash
# 设置代理环境变量
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# 然后运行构建
make release
```

**方案B: 配置Docker代理**
```bash
# 创建Docker代理配置
mkdir -p ~/.docker
cat > ~/.docker/config.json << EOF
{
  "proxies": {
    "default": {
      "httpProxy": "http://127.0.0.1:7890",
      "httpsProxy": "http://127.0.0.1:7890",
      "noProxy": "localhost,127.0.0.1"
    }
  }
}
EOF
```

**方案C: 使用镜像源**
修改`.bazelrc`文件，添加镜像源:
```
build --registry=https://mirrors.aliyun.com/bazel
```

### 3. Python环境问题

#### 问题4: Python依赖安装失败
**错误信息**:
```
ModuleNotFoundError: No module named 'loguru'
```

**原因**: Python虚拟环境未正确设置或依赖未安装。

**解决方案**:
```bash
# 1. 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装Python依赖
cd python
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 3. 设置PYTHONPATH
export PYTHONPATH=/path/to/primihub/python:$PYTHONPATH
```

#### 问题5: Python版本不兼容
**错误信息**:
```
python version must > 3.6
```

**解决方案**:
```bash
# 检查Python版本
python3 --version

# 如果版本过低，安装新版本
# Ubuntu/Debian
sudo apt-get install python3.8 python3.8-dev

# 使用特定版本
PYTHON_BIN=python3.8 ./pre_build.sh
```

### 4. 服务启动问题

#### 问题6: 端口被占用
**错误信息**:
```
Address already in use
```

**解决方案**:
```bash
# 1. 停止现有服务
./stop_server.sh

# 2. 检查端口占用
netstat -tlnp | grep :50050

# 3. 修改配置文件中的端口
# 编辑 config/node0.yaml, config/node1.yaml, config/node2.yaml
# 修改 grpc_port 和 proxy_server.port
```

#### 问题7: Meta服务启动失败
**错误信息**:
```
waiting for meta server start..., it will cost about 6 seconds
```

**解决方案**:
```bash
# 1. 检查Java是否安装
java -version

# 2. 安装Java 8（如果未安装）
# Ubuntu/Debian
sudo apt-get install openjdk-8-jre

# 3. 手动下载meta服务
cd meta_service
wget https://primihub.oss-cn-beijing.aliyuncs.com/tools/meta_service_v1.tar.gz
tar -zxf meta_service_v1.tar.gz
```

### 4.5. Docker 镜像构建问题

#### 问题7.1: 无法拉取基础镜像 primihub/primihub-base
**错误信息**:
```
ERROR: failed to resolve source metadata for docker.io/primihub/primihub-base:latest:
failed to do request: Head "https://registry-1.docker.io/v2/primihub/primihub-base/manifests/latest":
dial tcp 108.160.170.43:443: i/o timeout
```

**原因**:
- Docker Hub 网络连接超时
- `primihub/primihub-base` 镜像不存在或无法访问
- Docker daemon 代理配置未生效

**解决方案 (推荐)**: 使用现有版本镜像作为基础

1. **登录阿里云镜像仓库**:
```bash
echo "your_password" | docker login --username=primihub --password-stdin registry.cn-beijing.aliyuncs.com
```

2. **创建 Dockerfile.build**:
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

3. **构建镜像**:
```bash
docker build -t primihub/primihub-node:1.8.0 -f Dockerfile.build .
```

**优势**:
- ✅ 基于现有镜像增量更新，速度快（约 40 秒）
- ✅ 避免基础镜像拉取问题
- ✅ 减少网络依赖
- ✅ 镜像大小: 4.58GB (磁盘), 1.01GB (内容)

**替代方案**: 配置 Docker 镜像加速器

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

重启 Docker:
```bash
sudo systemctl restart docker
```

#### 问题7.2: Docker 构建时 apt/pip 安装失败
**错误信息**:
```
E: Failed to fetch http://archive.ubuntu.com/ubuntu/...
```

**解决方案**:
```bash
# 方案1: 使用国内镜像源
# 在 Dockerfile 中添加:
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# 方案2: 使用代理
# 构建时传递代理参数:
docker build --build-arg http_proxy=http://127.0.0.1:7890 \
             --build-arg https_proxy=http://127.0.0.1:7890 \
             -t primihub/primihub-node:1.8.0 .
```

#### 问题7.3: 构建产物不完整
**错误信息**:
```
ADD failed: file not found in build context
```

**解决方案**:
```bash
# 1. 确认构建产物存在
ls -lh bazel-bin.tar.gz commit.txt

# 2. 检查 tar 包内容
tar -tzf bazel-bin.tar.gz | head -20

# 3. 重新编译和打包
make release mysql=y
tar zcf bazel-bin.tar.gz \
    bazel-bin/cli bazel-bin/node bazel-bin/task_main \
    bazel-bin/src/primihub/pybind_warpper/*.so \
    bazel-bin/src/primihub/task/pybind_wrapper/*.so \
    python config example data
```

### 5. 运行时问题

#### 问题8: CLI连接失败
**错误信息**:
```
failed to connect to all addresses retry times: 0
```

**解决方案**:
```bash
# 1. 检查服务是否启动
ps aux | grep -E "(node|fusion)" | grep -v grep

# 2. 检查端口监听
netstat -tln | grep :50050

# 3. 检查配置文件中的IP地址
# 确保 config/*.yaml 中的 location 和 proxy_server.ip 正确
# 通常应为 "127.0.0.1" 或 "localhost"
```

#### 问题9: 任务执行超时
**错误信息**:
```
task timeout
```

**解决方案**:
```bash
# 1. 增加超时时间
timeout 120 ./primihub-cli --task_config_file="example/..."

# 2. 检查数据文件
ls -la data/*.csv

# 3. 查看详细日志
tail -f log_node0
```

## 构建优化建议

### 1. 并行构建加速
```bash
# 使用多线程构建（根据CPU核心数调整）
make release jobs=4
```

### 2. 选择性构建
```bash
# 仅构建必要组件
make release mysql=  # 跳过MySQL支持
```

### 3. 调试构建
```bash
# 启用调试信息
make release debug=y

# 启用详细输出
make release verbose=y
```

## 环境配置检查清单

在开始构建前，请检查以下项目:

### 系统依赖
- [ ] m4 工具
- [ ] gcc/g++ 编译器
- [ ] make 构建工具
- [ ] libmysqlclient-dev (如需MySQL支持)
- [ ] Java 8+ (用于meta服务)

### Python环境
- [ ] Python 3.6+
- [ ] python3.x-dev 头文件
- [ ] virtualenv (推荐)
- [ ] pip 包管理器

### 网络配置
- [ ] 可访问外网或配置代理
- [ ] Docker镜像源配置（如使用Docker）
- [ ] pip镜像源配置

### 权限检查
- [ ] 有sudo权限安装系统包
- [ ] 有权限创建目录和文件
- [ ] 有权限监听端口（50050+）

## 快速诊断命令

```bash
# 1. 检查系统依赖
which m4 gcc g++ make java python3

# 2. 检查Python环境
python3 --version
python3-config --includes

# 3. 检查网络连接
curl -I https://github.com
curl -I https://registry-1.docker.io

# 4. 检查服务状态
./stop_server.sh  # 停止所有服务
./start_server.sh # 启动服务
ps aux | grep -E "(node|fusion)" | grep -v grep

# 5. 测试基本功能
./primihub-cli --version
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"
```

## 常见构建命令示例

### 标准构建流程
```bash
# 1. 准备环境
./pre_build.sh

# 2. 构建项目
make release mysql=y jobs=4

# 3. 启动服务
./start_server.sh

# 4. 运行测试
./primihub-cli --task_config_file="example/psi_ecdh_task_conf.json"

# 5. 验证结果
cat data/result/psi_result.csv
```

### 开发调试流程
```bash
# 1. 清理环境
make clean
./stop_server.sh

# 2. 调试构建
make release debug=y verbose=y

# 3. 查看构建日志
tail -f bazel-out/linux_x86_64-fastbuild/bin/cli.log

# 4. 运行测试并查看详细日志
GLOG_v=3 ./primihub-cli --task_config_file="example/..."
```

## 问题反馈

如果遇到本文档未覆盖的问题:

1. **查看详细日志**: 检查 `bazel-out/` 目录下的构建日志
2. **检查错误信息**: 复制完整的错误信息
3. **提供环境信息**:
   ```bash
   uname -a
   python3 --version
   bazel version
   lsb_release -a  # 或 cat /etc/os-release
   ```
4. **在GitHub提交Issue**: https://github.com/primihub/primihub/issues

## 版本历史

- **v1.0** (2026-01-01): 初始版本，基于实际测试经验整理
- **v1.1** (2026-01-01): 添加网络代理配置和Python环境问题
- **v2.0** (2026-01-01): 整合到构建脚本和Makefile中
- **v2.1** (2026-01-09): 添加 Docker 镜像构建问题排查，包括基础镜像拉取失败、增量构建方案等

---
*最后更新: 2026-01-09*
*基于 PrimiHub 实际构建测试经验*