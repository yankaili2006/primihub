#!/usr/bin/env python3
"""
测试PrimiHub基本功能
"""

import sys
import os
import pandas as pd

# 添加Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

print("="*60)
print("PrimiHub 功能测试")
print("="*60)

# 1. 测试数据读取
print("\n1. 测试数据读取功能")
test_files = [
    "data/test_party_0.csv",
    "data/test_party_1.csv", 
    "data/test_party_2.csv",
    "data/train_party_0.csv",
    "data/train_party_1.csv",
    "data/train_party_2.csv"
]

for file in test_files:
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            print(f"  ✓ {file}: {df.shape[0]}行, {df.shape[1]}列")
        except Exception as e:
            print(f"  ✗ {file}: 读取失败 - {e}")
    else:
        print(f"  ⚠ {file}: 文件不存在")

# 2. 测试配置文件
print("\n2. 测试配置文件")
config_files = [
    "config/node0.yaml",
    "config/node1.yaml",
    "config/node2.yaml",
    "config/primihub_node0.yaml",
    "config/primihub_node1.yaml",
    "config/primihub_node2.yaml"
]

for file in config_files:
    if os.path.exists(file):
        print(f"  ✓ {file}: 存在")
    else:
        print(f"  ⚠ {file}: 不存在")

# 3. 测试示例任务配置
print("\n3. 测试示例任务配置")
example_files = [
    "example/psi_ecdh_task_conf.json",
    "example/psi_kkrt_task_conf.json",
    "example/mpc_lr_task_conf.json",
    "example/python_code.json"
]

for file in example_files:
    if os.path.exists(file):
        print(f"  ✓ {file}: 存在")
    else:
        print(f"  ⚠ {file}: 不存在")

# 4. 检查Python模块
print("\n4. 检查Python模块结构")
try:
    import primihub
    print("  ✓ primihub 包可导入")
    
    # 列出子模块
    import inspect
    import primihub as ph
    
    modules = []
    for name in dir(ph):
        if not name.startswith('_'):
            try:
                obj = getattr(ph, name)
                if inspect.ismodule(obj):
                    modules.append(name)
            except:
                pass
    
    print(f"  找到 {len(modules)} 个子模块: {', '.join(modules[:5])}{'...' if len(modules) > 5 else ''}")
    
except ImportError as e:
    print(f"  ✗ primihub 导入失败: {e}")

print("\n" + "="*60)
print("测试完成")
print("\n建议:")
print("1. 要运行完整系统，需要解决网络问题编译C++部分")
print("2. 或使用Docker: docker-compose up -d")
print("3. 或从GitHub Releases下载预编译版本")
print("4. Python SDK可用于开发和测试算法")
print("="*60)