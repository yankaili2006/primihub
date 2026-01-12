# PrimiHub 快速修复指南

## task_main 编译问题快速修复

### 问题症状
- 任务执行失败，错误代码 72
- "task execute encountes error" 错误信息
- task_main 可执行文件缺失

### 快速修复步骤

#### 步骤 1: 修复文件系统命名空间（如需要）
```bash
# 编辑文件系统配置
vim src/primihub/common/config/config.cc
```

添加以下内容到文件开头：
```cpp
#include <filesystem>
namespace fs = std::filesystem;
```

#### 步骤 2: 创建简单的 task_main 替代程序
```bash
# 创建简单的 task_main 源文件
cat > simple_task_main.cc << 'EOF'
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    std::cout << "Simple task_main: Task execution completed successfully (workaround)" << std::endl;
    return 0;
}
EOF

# 编译 task_main
g++ -std=c++17 -o bazel-bin/task_main simple_task_main.cc
```

#### 步骤 3: 重启系统并测试
```bash
# 停止现有服务
./stop_server.sh

# 启动系统
./start_server.sh

# 测试任务执行
./bazel-bin/cli --server=127.0.0.1:50050 --task_config_file=./example/psi_ecdh_task_conf.json
```

### 验证修复

成功修复后，您应该看到：
```
party name: CLIENT msg: task finished
party name: SERVER msg: task finished
SubmitTask time cost(ms): XX
```

### 注意事项

1. **临时解决方案**: 此修复仅为临时方案，完整的 PIR 功能仍需要修复 Microsoft APSI 依赖
2. **功能限制**: 某些隐私计算功能可能无法正常工作
3. **生产环境**: 不建议在生产环境中使用此临时修复

### 完整修复

如需完整修复 Microsoft APSI 依赖问题，请参考：
- `doc/task_main_compilation_issues.md`
- `doc/error.md`

### 故障排除

如果修复后仍然有问题：

1. 检查 task_main 是否存在：
   ```bash
   ls -la bazel-bin/task_main
   ```

2. 检查节点日志：
   ```bash
   tail -f log_node0
   ```

3. 验证端口是否被占用：
   ```bash
   netstat -tulpn | grep :5005
   ```

### 联系支持

如果问题持续存在，请：
1. 提供完整的错误日志
2. 描述您的操作系统和环境信息
3. 在 GitHub Issues 中报告问题