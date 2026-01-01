#!/usr/bin/env python3
"""
验证PSI结果是否正确
"""

import pandas as pd
import sys

def load_csv(filepath):
    """加载CSV文件，处理可能的BOM字符"""
    try:
        # 尝试不同编码
        for encoding in ['utf-8', 'utf-8-sig', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"使用编码 {encoding} 成功读取 {filepath}")
                return df
            except UnicodeDecodeError:
                continue
        # 如果所有编码都失败，使用错误处理
        df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
        print(f"使用错误处理读取 {filepath}")
        return df
    except Exception as e:
        print(f"读取 {filepath} 失败: {e}")
        return None

def verify_psi_result():
    """验证PSI结果"""
    print("=" * 60)
    print("PSI结果验证")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n1. 加载原始数据...")
    client_df = load_csv("data/client_e.csv")
    server_df = load_csv("data/server_e.csv")
    psi_result_df = load_csv("data/result/psi_result.csv")
    
    if client_df is None or server_df is None or psi_result_df is None:
        print("数据加载失败")
        return False
    
    print(f"\n客户端数据: {len(client_df)} 行, {len(client_df.columns)} 列")
    print(f"服务端数据: {len(server_df)} 行, {len(server_df.columns)} 列")
    print(f"PSI结果: {len(psi_result_df)} 行, {len(psi_result_df.columns)} 列")
    
    # 2. 检查列名
    print("\n2. 检查列名...")
    print(f"客户端列: {list(client_df.columns)}")
    print(f"服务端列: {list(server_df.columns)}")
    print(f"PSI结果列: {list(psi_result_df.columns)}")
    
    # 3. 提取ID列
    # 处理可能的列名问题（BOM字符）
    client_id_col = None
    server_id_col = None
    psi_id_col = None
    
    for col in client_df.columns:
        if 'id' in col.lower():
            client_id_col = col
            break
    
    for col in server_df.columns:
        if 'id' in col.lower():
            server_id_col = col
            break
    
    for col in psi_result_df.columns:
        if 'id' in col.lower() or col.strip().startswith('\ufeff'):
            psi_id_col = col
            break
    
    if not client_id_col:
        client_id_col = client_df.columns[0]
    if not server_id_col:
        server_id_col = server_df.columns[0]
    if not psi_id_col:
        psi_id_col = psi_result_df.columns[0]
    
    print(f"\n使用的ID列:")
    print(f"  客户端: {client_id_col}")
    print(f"  服务端: {server_id_col}")
    print(f"  PSI结果: {psi_id_col}")
    
    # 4. 计算理论交集
    print("\n3. 计算理论交集...")
    client_ids = set(client_df[client_id_col].astype(str).str.strip())
    server_ids = set(server_df[server_id_col].astype(str).str.strip())
    
    print(f"客户端ID数量: {len(client_ids)}")
    print(f"服务端ID数量: {len(server_ids)}")
    
    theoretical_intersection = client_ids.intersection(server_ids)
    print(f"理论交集数量: {len(theoretical_intersection)}")
    
    # 5. 获取实际PSI结果
    print("\n4. 获取实际PSI结果...")
    # 清理PSI结果中的ID（去除BOM字符和空格）
    psi_ids = set()
    for id_val in psi_result_df[psi_id_col]:
        if pd.isna(id_val):
            continue
        # 去除BOM字符和空格
        clean_id = str(id_val).strip()
        # 处理可能的BOM字符
        if clean_id.startswith('\ufeff'):
            clean_id = clean_id[1:]
        psi_ids.add(clean_id)
    
    print(f"实际PSI结果数量: {len(psi_ids)}")
    
    # 6. 验证结果
    print("\n5. 验证结果...")
    
    # 检查PSI结果是否都是理论交集的子集
    missing_in_psi = theoretical_intersection - psi_ids
    extra_in_psi = psi_ids - theoretical_intersection
    
    if len(missing_in_psi) == 0 and len(extra_in_psi) == 0:
        print("✅ PSI结果完全正确！")
        print(f"  交集数量: {len(theoretical_intersection)}")
        return True
    else:
        print("❌ PSI结果有问题！")
        
        if len(missing_in_psi) > 0:
            print(f"  缺失的ID ({len(missing_in_psi)}个): {sorted(list(missing_in_psi))[:10]}{'...' if len(missing_in_psi) > 10 else ''}")
        
        if len(extra_in_psi) > 0:
            print(f"  多余的ID ({len(extra_in_psi)}个): {sorted(list(extra_in_psi))[:10]}{'...' if len(extra_in_psi) > 10 else ''}")
        
        # 检查是否是大小写或格式问题
        print("\n6. 详细检查...")
        
        # 检查客户端所有ID是否都在服务端中
        client_only = client_ids - server_ids
        if len(client_only) > 0:
            print(f"  客户端独有的ID ({len(client_only)}个): {sorted(list(client_only))[:5]}{'...' if len(client_only) > 5 else ''}")
        else:
            print("  所有客户端ID都在服务端中")
        
        # 显示前几个ID的对比
        print("\n7. 样本对比:")
        print("  客户端前5个ID:", sorted(list(client_ids))[:5])
        print("  服务端前5个ID:", sorted(list(server_ids))[:5])
        print("  PSI结果前5个ID:", sorted(list(psi_ids))[:5])
        
        return False

def manual_verification():
    """手动验证"""
    print("\n" + "=" * 60)
    print("手动验证")
    print("=" * 60)
    
    # 读取原始文件内容
    with open("data/client_e.csv", 'r', encoding='utf-8-sig') as f:
        client_lines = f.readlines()
    
    with open("data/server_e.csv", 'r', encoding='utf-8-sig') as f:
        server_lines = f.readlines()
    
    with open("data/result/psi_result.csv", 'r', encoding='utf-8-sig') as f:
        psi_lines = f.readlines()
    
    # 提取ID（跳过标题行）
    client_ids = []
    for line in client_lines[1:]:  # 跳过标题
        parts = line.strip().split(',')
        if parts:
            client_ids.append(parts[0].strip())
    
    server_ids = []
    for line in server_lines[1:]:  # 跳过标题
        parts = line.strip().split(',')
        if parts:
            server_ids.append(parts[0].strip())
    
    psi_ids = []
    for line in psi_lines[1:]:  # 跳过标题
        parts = line.strip().split(',')
        if parts:
            psi_ids.append(parts[0].strip())
    
    print(f"客户端ID数量: {len(client_ids)}")
    print(f"服务端ID数量: {len(server_ids)}")
    print(f"PSI结果数量: {len(psi_ids)}")
    
    # 转换为集合
    client_set = set(client_ids)
    server_set = set(server_ids)
    psi_set = set(psi_ids)
    
    # 计算交集
    intersection = client_set.intersection(server_set)
    
    print(f"\n理论交集: {len(intersection)}")
    print(f"实际PSI结果: {len(psi_set)}")
    
    if intersection == psi_set:
        print("✅ 手动验证通过！")
        print(f"交集ID: {sorted(list(intersection))}")
    else:
        print("❌ 手动验证失败！")
        
        missing = intersection - psi_set
        extra = psi_set - intersection
        
        if missing:
            print(f"缺失的ID: {sorted(list(missing))}")
        
        if extra:
            print(f"多余的ID: {sorted(list(extra))}")

def main():
    """主函数"""
    print("PSI结果验证工具")
    print("=" * 60)
    
    # 运行自动验证
    auto_result = verify_psi_result()
    
    # 运行手动验证
    manual_verification()
    
    print("\n" + "=" * 60)
    if auto_result:
        print("总结: ✅ PSI结果验证通过")
    else:
        print("总结: ❌ PSI结果验证失败")
    print("=" * 60)

if __name__ == "__main__":
    main()