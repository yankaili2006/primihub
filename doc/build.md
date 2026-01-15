# Build

## Choice PIR support
 
 Use microsoft-apsi definition in `.bazelrc` file.

if use microsoft-apsi PIR support, use

```
build:linux --define microsoft-apsi=true
```

or use openminded PIR library default.

### Microsoft APSI dependencies that need to be installed manually
***It is recommended to install with homebrew***
 
 * flatbuffers 2.0.8
 * tclap 1.2.5
 * zeromq 4.3.4
 * cppzmq 4.8.1

```
brew install flatbuffers tclap zeromq cppzmq
```

## 常见编译问题

### task_main 编译失败

#### 问题描述
编译过程中 `task_main` 可执行文件无法构建，导致任务执行失败。

#### 主要错误
1. **Microsoft APSI 依赖错误**:
   ```
   error: 'digit_t' undeclared
   ```

2. **C++17 兼容性错误**:
   ```
   error: 'string_view' is not a member of 'std'
   ```

#### 临时解决方案
1. **使用 THREAD 模式**:
   修改 `src/primihub/node/worker/worker.h` 中的任务执行模式

2. **创建简单的 task_main**:
   ```bash
   g++ -std=c++17 -o bazel-bin/task_main simple_task_main.cc
   ```

#### 完整解决方案
参考 `doc/task_main_compilation_issues.md` 获取详细修复步骤。

### 文件系统命名空间错误

#### 问题描述
```
error: 'fs' has not been declared
```

#### 解决方案
修改 `src/primihub/common/config/config.cc`:
```cpp
#include <filesystem>
namespace fs = std::filesystem;
```

### 端口绑定错误

#### 问题描述
```
Address already in use
```

#### 解决方案
1. 停止现有服务:
   ```bash
   ./stop_server.sh
   ```

2. 检查并杀死残留进程:
   ```bash
   pkill -f "primihub-node"
   ```

## 验证构建

构建完成后验证关键组件:

```bash
# 检查生成的可执行文件
ls -la bazel-bin/

# 验证节点程序
./bazel-bin/node --help

# 验证 CLI 工具
./bazel-bin/cli --help

# 启动完整系统测试
./start_server.sh
```

## 故障排除

如果遇到编译问题:

1. 清理构建缓存:
   ```bash
   bazel clean
   ```

2. 使用详细模式构建:
   ```bash
   bazel build --verbose_failures //:node
   ```

3. 检查详细错误日志

更多问题参考 `doc/error.md` 和 `doc/task_main_compilation_issues.md`。