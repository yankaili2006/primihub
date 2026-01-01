#!/usr/bin/env python3
"""
简单的PrimiHub测试脚本
"""

import sys
import os

# 添加Python SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

try:
    # 测试基本导入
    print("1. 测试基本导入...")
    import primihub
    print("   ✓ primihub 导入成功")
    
    # 测试上下文
    print("\n2. 测试上下文模块...")
    from primihub.context import Context
    print("   ✓ Context 导入成功")
    
    # 测试日志工具
    print("\n3. 测试日志工具...")
    try:
        from primihub.utils.logger_util import logger
        print("   ⚠ logger_util 需要loguru依赖")
    except ImportError as e:
        print(f"   ⚠ logger_util导入失败: {e}")
    
    # 测试数据模块
    print("\n4. 测试数据模块...")
    try:
        from primihub.data import data_utils
        print("   ✓ data_utils 导入成功")
    except ImportError as e:
        print(f"   ⚠ data_utils导入失败: {e}")
    
    # 测试FL模块
    print("\n5. 测试联邦学习模块...")
    try:
        from primihub.FL import LogisticRegression
        print("   ✓ LogisticRegression 导入成功")
    except ImportError as e:
        print(f"   ⚠ FL模块导入失败: {e}")
    
    print("\n" + "="*50)
    print("总结: Python SDK基本结构完整，但需要安装依赖")
    print("需要安装的依赖:")
    print("  pip install loguru pandas numpy scikit-learn")
    print("\n要运行完整示例，需要:")
    print("  1. 安装Python依赖")
    print("  2. 编译C++部分或使用Docker")
    print("  3. 启动meta服务和node服务")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()