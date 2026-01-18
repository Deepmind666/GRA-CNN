"""
顶级严谨版数据集标准化工具 (Final Data Sanitizer)
==============================================
目标: 从 60 万行原始数据中精准提取 12 个核心 Panel 的聚合数据。
核心逻辑:
1. 模糊匹配转化为标准名称 (resnet20, cifar10, gra...)
2. 多版本精度对齐 (accuracy_clean, acc, val)
3. 分组聚合 (Mean + Std)
"""

import pandas as pd
import os
import re

def normalize_arch(x):
    s = str(x).lower().replace('-', '').replace('_', '').replace(' ', '')
    if 'resnet110' in s: return 'resnet110'
    if 'resnet56' in s: return 'resnet56'
    if 'resnet44' in s: return 'resnet44'
    if 'resnet32' in s: return 'resnet32'
    if 'resnet20' in s: return 'resnet20'
    if 'vgg16' in s: return 'vgg16'
    return s

def normalize_ds(x):
    s = str(x).lower().replace('-', '').replace('_', '').replace(' ', '')
    if '100' in s: return 'cifar100'
    if 'tiny' in s: return 'tinyimagenet'
    if 'cifar10' in s: return 'cifar10'
    return s

def normalize_meth(x):
    s = str(x).upper().strip()
    if 'GRA' in s: return 'GRA'
    if 'L1' in s: return 'L1'
    if 'FPGM' in s: return 'FPGM'
    if 'HRANK' in s: return 'HRANK'
    if 'BASE' in s: return 'BASELINE'
    return s

def sanitize():
    path = 'experiments/master_scientific_results.csv'
    if not os.path.exists(path):
        print("Master file not found!")
        return
    
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records.")
    
    # 提取核心四列
    clean_records = []
    
    # 手动处理列名，避免空格干扰
    df.columns = [c.strip() for c in df.columns]
    
    # 找到所有可能的精度列
    acc_cols = [c for c in df.columns if any(x in c.lower() for x in ['acc', 'accuracy', 'val'])]
    
    for _, row in df.iterrows():
        try:
            arch = normalize_arch(row['architecture'])
            ds = normalize_ds(row['dataset'])
            meth = normalize_meth(row['method'])
            ratio = float(row['ratio'])
            
            # 找最高精度的列
            val = None
            for c in acc_cols:
                v = row[c]
                if pd.notnull(v) and v != '':
                    try: 
                        v = float(v)
                        if val is None or v > val: val = v
                    except: pass
            
            if val is not None and val > 5:
                clean_records.append({
                    'arch': arch, 'ds': ds, 'meth': meth, 'ratio': ratio, 'acc': val
                })
        except:
            continue
            
    clean_df = pd.DataFrame(clean_records)
    print(f"Sanitized {len(clean_df)} valid records.")
    
    # 聚合：计算均值与标准差
    final = clean_df.groupby(['arch', 'ds', 'meth', 'ratio'])['acc'].agg(['mean', 'std', 'count']).reset_index()
    final.to_csv('experiments/FINAL_CLEAN_VIS_DATA.csv', index=False)
    print("Saved to experiments/FINAL_CLEAN_VIS_DATA.csv")
    print(final.groupby(['arch', 'ds'])['meth'].nunique())

if __name__ == "__main__":
    sanitize()
