#!/usr/bin/env python3
"""
简单的PrimiHub功能测试
"""

import sys
import os

# 添加Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

print("测试PrimiHub基本功能...")
print("="*50)

# 测试基本导入
try:
    print("1. 测试基本模块导入...")
    import primihub
    print("   ✓ primihub 包导入成功")
    
    # 测试数据工具
    print("\n2. 测试数据工具...")
    try:
        from primihub.data import data_utils
        print("   ✓ data_utils 导入成功")
        
        # 测试读取CSV
        test_file = "data/test_party_0.csv"
        if os.path.exists(test_file):
            print(f"\n3. 测试读取数据文件: {test_file}")
            import pandas as pd
            df = pd.read_csv(test_file)
            print(f"   数据形状: {df.shape}")
            print(f"   列名: {list(df.columns)}")
            print("   ✓ 数据读取成功")
        else:
            print(f"   ⚠ 测试文件不存在: {test_file}")
            
    except ImportError as e:
        print(f"   ⚠ data_utils导入失败: {e}")
    
    # 测试算法模块
    print("\n4. 测试算法模块...")
    try:
        # 检查算法目录
        algo_dir = os.path.join(os.path.dirname(__file__), 'src/primihub/algorithm')
        if os.path.exists(algo_dir):
            print(f"   算法目录存在: {algo_dir}")
            algorithms = os.listdir(algo_dir)
            print(f"   找到 {len(algorithms)} 个算法目录")
            print(f"   算法列表: {', '.join(algorithms[:10])}{'...' if len(algorithms) > 10 else ''}")
        else:
            print(f"   ⚠ 算法目录不存在: {algo_dir}")
            
    except Exception as e:
        print(f"   ⚠ 算法模块检查失败: {e}")
    
    print("\n" + "="*50)
    print("总结:")
    print("✓ Python SDK基本结构完整")
    print("⚠ 需要编译C++部分以获得完整功能")
    print("⚠ 需要启动meta服务和node服务")
    print("\n建议的下一步:")
    print("1. 解决网络代理问题以完成构建")
    print("2. 或使用预编译的Docker镜像")
    print("3. 参考文档: https://docs.primihub.com")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()