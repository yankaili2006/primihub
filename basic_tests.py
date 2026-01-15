#!/usr/bin/env python3
"""
PrimiHub 基本功能测试
基于文档和代码分析设计的基础测试
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# 添加Python SDK路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def test_data_files():
    """测试数据文件是否存在且可读"""
    print("=== 测试数据文件 ===")
    
    test_files = [
        "data/client_e.csv",
        "data/server_e.csv", 
        "data/test_party_0.csv",
        "data/test_party_1.csv",
        "data/test_party_2.csv",
        "data/train_party_0.csv",
        "data/train_party_1.csv",
        "data/train_party_2.csv",
        "data/mpc_test.csv"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                print(f"✓ {file}: {len(df)}行, {len(df.columns)}列")
            except Exception as e:
                print(f"✗ {file}: 读取失败 - {e}")
        else:
            print(f"✗ {file}: 文件不存在")

def test_config_files():
    """测试配置文件"""
    print("\n=== 测试配置文件 ===")
    
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
            with open(file, 'r') as f:
                content = f.read()
                lines = len(content.split('\n'))
                print(f"✓ {file}: {lines}行")
        else:
            print(f"✗ {file}: 文件不存在")

def test_task_configs():
    """测试任务配置文件"""
    print("\n=== 测试任务配置 ===")
    
    task_configs = [
        "example/psi_ecdh_task_conf.json",
        "example/psi_kkrt_task_conf.json",
        "example/mpc_lr_task_conf.json",
        "example/mpc_statistics_sum_task_conf.json",
        "example/python_code.json"
    ]
    
    for config in task_configs:
        if os.path.exists(config):
            try:
                with open(config, 'r') as f:
                    task = json.load(f)
                    task_type = task.get('task_type', '未知')
                    task_name = task.get('task_name', '未知')
                    print(f"✓ {config}: {task_type} - {task_name}")
            except Exception as e:
                print(f"✗ {config}: 解析失败 - {e}")
        else:
            print(f"✗ {config}: 文件不存在")

def test_psi_data():
    """测试PSI数据"""
    print("\n=== 测试PSI数据 ===")
    
    client_data = "data/client_e.csv"
    server_data = "data/server_e.csv"
    
    if os.path.exists(client_data) and os.path.exists(server_data):
        client_df = pd.read_csv(client_data)
        server_df = pd.read_csv(server_data)
        
        print(f"客户端数据: {len(client_df)}行, {len(client_df.columns)}列")
        print(f"服务端数据: {len(server_df)}行, {len(server_df.columns)}列")
        
        # 检查是否有共同列
        client_cols = set(client_df.columns)
        server_cols = set(server_df.columns)
        common_cols = client_cols.intersection(server_cols)
        
        if common_cols:
            print(f"共同列: {', '.join(common_cols)}")
            
            # 检查是否有交集数据
            if 'id' in common_cols:
                client_ids = set(client_df['id'].dropna())
                server_ids = set(server_df['id'].dropna())
                intersection = client_ids.intersection(server_ids)
                print(f"ID交集数量: {len(intersection)}")
        else:
            print("警告: 没有共同列")
    else:
        print("PSI数据文件不存在")

def test_mpc_data():
    """测试MPC数据"""
    print("\n=== 测试MPC数据 ===")
    
    # 检查MPC测试数据
    mpc_files = [
        "data/mpc_test.csv",
        "data/mpc_arithmetic_0.csv",
        "data/mpc_arithmetic_1.csv", 
        "data/mpc_arithmetic_2.csv"
    ]
    
    for file in mpc_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            print(f"{file}:")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            print(f"  数据类型: {df.dtypes.to_dict()}")
            
            # 显示统计信息
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  数值列统计:")
                for col in numeric_cols[:3]:  # 只显示前3列
                    print(f"    {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
        else:
            print(f"{file}: 文件不存在")

def test_python_sdk():
    """测试Python SDK"""
    print("\n=== 测试Python SDK ===")
    
    try:
        from primihub.context import Context
        print("✓ primihub.context 导入成功")
    except ImportError as e:
        print(f"✗ primihub.context 导入失败: {e}")
    
    try:
        from primihub.utils import logger_util
        print("✓ primihub.utils.logger_util 导入成功")
    except ImportError as e:
        print(f"✗ primihub.utils.logger_util 导入失败: {e}")
    
    # 检查算法目录
    algorithm_dir = "src/primihub/algorithm"
    if os.path.exists(algorithm_dir):
        algorithms = [f for f in os.listdir(algorithm_dir) if os.path.isdir(os.path.join(algorithm_dir, f))]
        print(f"✓ 找到 {len(algorithms)} 个算法目录")
        print(f"  算法: {', '.join(algorithms[:10])}{'...' if len(algorithms) > 10 else ''}")
    else:
        print(f"✗ 算法目录不存在: {algorithm_dir}")

def test_simple_mpc_example():
    """测试简单的MPC计算示例"""
    print("\n=== 简单MPC计算示例 ===")
    
    # 读取MPC测试数据
    mpc_file = "data/mpc_test.csv"
    if os.path.exists(mpc_file):
        df = pd.read_csv(mpc_file)
        
        print("数据预览:")
        print(df.head())
        
        print("\n基本统计:")
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        
        # 计算每列的和（模拟MPC统计）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("\n数值列求和（模拟MPC计算）:")
            for col in numeric_cols:
                col_sum = df[col].sum()
                print(f"  {col}: {col_sum:.2f}")
                
            # 计算平均值
            print("\n数值列平均值:")
            for col in numeric_cols:
                col_mean = df[col].mean()
                print(f"  {col}: {col_mean:.2f}")
    else:
        print("MPC测试数据文件不存在")

def main():
    """主测试函数"""
    print("=" * 60)
    print("PrimiHub 基本功能测试")
    print("=" * 60)
    
    # 运行所有测试
    test_data_files()
    test_config_files()
    test_task_configs()
    test_psi_data()
    test_mpc_data()
    test_python_sdk()
    test_simple_mpc_example()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    print("\n建议的下一步:")
    print("1. 确保所有服务正在运行: ./start_server.sh")
    print("2. 运行PSI测试: ./primihub-cli --task_config_file=\"example/psi_ecdh_task_conf.json\"")
    print("3. 运行MPC逻辑回归测试: ./primihub-cli --task_config_file=\"example/mpc_lr_task_conf.json\"")
    print("4. 查看结果文件: cat data/result/psi_result.csv")
    print("5. 停止服务: ./stop_server.sh")

if __name__ == "__main__":
    main()