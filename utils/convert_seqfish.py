# -*- coding: utf-8 -*-
"""
将 seqFISH+ h5ad 数据转换为 STdGCN 标准输入格式。

该脚本面向一次性数据准备任务：读取单细胞与空间 h5ad，输出 STdGCN
训练流程需要的四个基础文件。由于 seqFISH+ 单细胞文件较大，日志会明确
提示加载阶段可能耗时。
"""
import scanpy as sc
import pandas as pd
import os

# 1. 读取原始 h5ad 文件。单细胞参考通常体积较大，加载阶段可能明显耗时。
print("[信息] 加载 seqFISH+ 单细胞数据（约3.6GB，请耐心等待）...")
sc_adata = sc.read_h5ad('data/seqfish+/seqfish+_sc_adata.h5ad')
print(f"  单细胞数据维度: {sc_adata.shape}")

print("[信息] 加载 seqFISH+ 空间数据...")
st_adata = sc.read_h5ad('data/seqfish+/seqfish+_st_adata.h5ad')
print(f"  空间数据维度: {st_adata.shape}")

# 2. 创建输出目录，保持与其他转换脚本一致的文件布局。
output_dir = 'data/seqfish_tsv'
os.makedirs(output_dir, exist_ok=True)

# 3. 保存单细胞表达矩阵。稀疏矩阵转稠密后再写 TSV，保证下游 pandas/scanpy 可读。
print("[信息] 保存 sc_data.tsv...")
sc_df = pd.DataFrame(
    sc_adata.X.toarray() if hasattr(sc_adata.X, 'toarray') else sc_adata.X,
    index=sc_adata.obs_names, 
    columns=sc_adata.var_names
)
sc_df.to_csv(f'{output_dir}/sc_data.tsv', sep='\t')

# 4. 保存单细胞标签。优先使用已知细胞类型列，避免把批次或样本元数据误当标签。
print("[信息] 保存 sc_label.tsv...")
cell_type_col = None
for col in ['cell_type', 'celltype', 'CellType', 'cluster', 'leiden', 'louvain', 'Cell_class']:
    if col in sc_adata.obs.columns:
        cell_type_col = col
        break

if cell_type_col is None:
    print(f"  可用的列: {sc_adata.obs.columns.tolist()}")
    cell_type_col = sc_adata.obs.columns[0]

print(f"  使用列 '{cell_type_col}' 作为细胞类型")
sc_label = pd.DataFrame({'cell': sc_adata.obs_names, 'cell_type': sc_adata.obs[cell_type_col].values})
sc_label.to_csv(f'{output_dir}/sc_label.tsv', sep='\t', index=False)

# 5. 保存空间表达矩阵，行索引必须与坐标文件保持同一套 spot/barcode。
print("[信息] 保存 ST_data.tsv...")
st_df = pd.DataFrame(
    st_adata.X.toarray() if hasattr(st_adata.X, 'toarray') else st_adata.X,
    index=st_adata.obs_names, 
    columns=st_adata.var_names
)
st_df.to_csv(f'{output_dir}/ST_data.tsv', sep='\t')

# 6. 保存空间坐标。优先采用 `obsm['spatial']`；缺失时再从 obs 中推断。
print("[信息] 保存 coordinates.csv...")
if 'spatial' in st_adata.obsm:
    coords = pd.DataFrame(st_adata.obsm['spatial'], index=st_adata.obs_names, columns=['x', 'y'])
else:
    # 尝试从 obs 中找坐标。
    x_col = y_col = None
    for col in st_adata.obs.columns:
        if 'x' in col.lower() and x_col is None:
            x_col = col
        if 'y' in col.lower() and y_col is None:
            y_col = col
    if x_col and y_col:
        coords = st_adata.obs[[x_col, y_col]].copy()
        coords.columns = ['x', 'y']
    else:
        raise ValueError("找不到空间坐标！")

coords.to_csv(f'{output_dir}/coordinates.csv')

print(f"\n[成功] 数据转换完成！")
print(f"文件保存到: {output_dir}/")
print(f"  - sc_data.tsv")
print(f"  - sc_label.tsv")
print(f"  - ST_data.tsv")
print(f"  - coordinates.csv")
