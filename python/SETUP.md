# PrimiHub Python环境配置指南

本文档说明如何正确配置PrimiHub的Python环境，特别是联邦学习(FL)功能所需的依赖。

## 重要更新（2026-01-02）

**⚠️ requirements.txt已过时**

原 `requirements.txt` 文件针对 Python 3.7-3.10 设计，使用 torch 1.13.1。该版本已不再可用，且与Python 3.12不兼容。

**✅ 推荐配置**

使用以下经过验证的配置：

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10-3.12 | **3.12.3已验证** |
| torch | **2.6.0+cpu** | ⚠️ 必须2.6+，否则FL失败 |
| torchvision | 0.21.0+cpu | 匹配torch版本 |
| opacus | 1.4.0+ | 差分隐私支持 |
| scikit-learn | 1.8.0+ | 机器学习库 |
| loguru | 0.7.3+ | 日志库 |
| phe | 1.5.0+ | Paillier同态加密 |

## 快速安装

### 方法1: 一键安装（推荐）

```bash
cd /path/to/primihub

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装PyTorch 2.6（CPU版本）
pip install --no-cache-dir \
  torch==2.6.0+cpu \
  torchvision==0.21.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install --no-cache-dir \
  loguru \
  scikit-learn \
  phe \
  opacus \
  numpy \
  pandas \
  pyarrow \
  grpcio \
  protobuf
```

### 方法2: 使用更新的requirements

创建 `requirements-2024.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu

# 核心依赖
torch==2.6.0+cpu
torchvision==0.21.0+cpu
numpy>=1.24.0
pandas>=1.5.0
scipy>=1.10.0

# 隐私计算
opacus==1.4.0
phe==1.5.0
tenseal==0.3.14; platform_machine != "arm64" and platform_machine != "aarch64"

# 机器学习
scikit-learn>=1.3.0

# 工具库
loguru
grpcio>=1.43.0
protobuf>=3.20.0,<4.0.0
pyarrow>=6.0.0

# 可选依赖
# ray==2.2.0
# transformers
# matplotlib
```

安装：
```bash
pip install -r requirements-2024.txt
```

## 验证安装

运行验证脚本：

```bash
python -c "
import sys
print(f'Python版本: {sys.version}')
print()

# 核心依赖
import torch
import torchvision
import numpy
import pandas
import scipy

print('✅ 核心依赖已安装')
print(f'  torch: {torch.__version__}')
print(f'  torchvision: {torchvision.__version__}')
print(f'  numpy: {numpy.__version__}')
print(f'  pandas: {pandas.__version__}')
print(f'  scipy: {scipy.__version__}')
print()

# FL相关依赖
import opacus
import sklearn
import loguru
import phe

print('✅ FL依赖已安装')
print(f'  opacus: {opacus.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  loguru: {loguru.__version__}')
print(f'  phe: {phe.__version__}')
print()

# 关键特性检查
print('✅ 关键特性检查')
print(f'  torch.nn.RMSNorm: {\"存在\" if hasattr(torch.nn, \"RMSNorm\") else \"缺失\"}')
print(f'  torch设备: {torch.device(\"cpu\")}')
print()

print('🎉 所有依赖验证通过！')
"
```

**预期输出**:
```
Python版本: 3.12.3 ...

✅ 核心依赖已安装
  torch: 2.6.0+cpu
  torchvision: 0.21.0+cpu
  numpy: 1.26.4
  pandas: 2.x.x
  scipy: 1.16.3

✅ FL依赖已安装
  opacus: 1.4.0
  scikit-learn: 1.8.0
  loguru: 0.7.3
  phe: 1.5.0

✅ 关键特性检查
  torch.nn.RMSNorm: 存在
  torch设备: cpu

🎉 所有依赖验证通过！
```

## 常见问题

### Q1: 为什么不能使用requirements.txt？

**A**: 原 `requirements.txt` 有以下问题：
1. 指定 `torch==1.13.1+cpu`，该版本已不再可用
2. `numpy==1.21.3` 不支持 Python 3.12
3. 许多包版本过旧，与新Python不兼容

### Q2: 为什么必须使用torch 2.6+？

**A**: torch 2.6+ 才有以下必需特性：
- `torch.nn.RMSNorm` (opacus需要)
- 修复了ONNX DiagnosticOptions导入问题
- 完整的Python 3.12支持

使用旧版本会导致：
```python
# torch 2.2.2
AttributeError: module 'torch.nn' has no attribute 'RMSNorm'

# torch 2.4.0
ImportError: cannot import name 'DiagnosticOptions' from 'torch.onnx._internal.exporter'
```

### Q3: 可以使用GPU版本的PyTorch吗？

**A**: 可以，如果有NVIDIA GPU：

```bash
# 检查CUDA版本
nvidia-smi

# 安装GPU版本 (CUDA 12.1示例)
pip install \
  torch==2.6.0+cu121 \
  torchvision==0.21.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

**注意**: GPU版本安装包很大（~2GB），确保有足够磁盘空间。

### Q4: 磁盘空间不足怎么办？

**A**:
```bash
# 1. 使用CPU版本（~200MB vs 2GB+）
pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. 使用--no-cache-dir避免缓存
pip install --no-cache-dir <package>

# 3. 清理pip缓存
pip cache purge
```

### Q5: 如何处理externally-managed-environment错误？

**A**: 使用虚拟环境或添加标志：

```bash
# 方法1: 虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate
pip install <package>

# 方法2: 用户安装
pip install --user <package>

# 方法3: 系统安装（不推荐）
pip install --break-system-packages <package>
```

## 测试FL功能

安装完依赖后，测试FL功能：

```bash
cd /path/to/primihub

# 测试横向联邦学习
./primihub-cli --task_config_file=example/FL/neural_network/hfl_binclass_plaintext.json
```

**预期结果**:
```
I20260102 06:57:05.483731 party name: Alice msg: task finished
I20260102 06:57:05.483739 party name: Bob msg: task finished
I20260102 06:57:05.483739 party name: Charlie msg: task finished
SubmitTask time cost(ms): 7851
```

**查看训练结果**:
```bash
# 查看训练指标
cat data/result/Bob_metrics.json

# 查看模型文件
ls -lh data/result/*_model.pkl
```

**训练性能**:
```json
{
  "train_acc": 0.9825,      // 准确率: 98.25%
  "train_f1": 0.9857,       // F1分数: 98.57%
  "train_auc": 0.9919       // AUC: 99.19%
}
```

## Docker环境

如果使用Docker，可以创建包含所有依赖的镜像：

```dockerfile
FROM python:3.12-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装PyTorch（CPU版本）
RUN pip install --no-cache-dir \
    torch==2.6.0+cpu \
    torchvision==0.21.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
RUN pip install --no-cache-dir \
    loguru \
    scikit-learn \
    phe \
    opacus \
    numpy \
    pandas \
    pyarrow \
    grpcio \
    protobuf

# 复制应用代码
COPY . /app

# 验证安装
RUN python -c "import torch; print(f'torch: {torch.__version__}')"

CMD ["/bin/bash"]
```

构建和运行：
```bash
docker build -t primihub-fl:latest .
docker run -it primihub-fl:latest
```

## 性能优化

### 使用国内PyPI镜像

编辑 `~/.pip/pip.conf`:
```ini
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
extra-index-url = https://download.pytorch.org/whl/cpu

[install]
trusted-host = mirrors.aliyun.com
```

### 并行安装

```bash
pip install --no-cache-dir --use-pep517 --upgrade pip setuptools wheel
pip install --no-cache-dir --prefer-binary <package>
```

## 故障排查

### 导入错误

```python
# 错误: ModuleNotFoundError
import torch  # ❌ ModuleNotFoundError: No module named 'torch'

# 检查pip安装位置
which pip
pip list | grep torch

# 检查Python路径
python -c "import sys; print(sys.path)"

# 确保使用正确的Python和pip
/path/to/venv/bin/python
/path/to/venv/bin/pip
```

### 版本冲突

```bash
# 查看已安装包
pip list

# 检查依赖树
pip install pipdeptree
pipdeptree

# 强制重新安装
pip install --force-reinstall --no-cache-dir torch==2.6.0+cpu
```

## 更多信息

- **官方文档**: https://docs.primihub.com
- **PyTorch文档**: https://pytorch.org/get-started/locally/
- **问题反馈**: https://github.com/primihub/primihub/issues

---

**文档更新日期**: 2026-01-02
**Python版本**: 3.10-3.12
**PyTorch版本**: 2.6.0+
