"""将 h5ad 格式转换为 STdGCN 所需的 TSV 格式脚本"""
import scanpy as sc
import pandas as pd
import os

# 加载数据
print("[信息] 正在加载 STARmap 单细胞数据...")
sc_adata = sc.read_h5ad('data/starmap/starmap_sc_adata.h5ad')
print(f"  Shape: {sc_adata.shape}")

print("[信息] 正在加载 STARmap 空间数据...")
st_adata = sc.read_h5ad('data/starmap/starmap_st_adata.h5ad')
print(f"  Shape: {st_adata.shape}")

# 创建输出目录
output_dir = 'data/starmap_tsv'
os.makedirs(output_dir, exist_ok=True)

# 保存单细胞表达矩阵
print("[信息] 正在保存 sc_data.tsv...")
sc_df = pd.DataFrame(sc_adata.X.toarray() if hasattr(sc_adata.X, 'toarray') else sc_adata.X,
                     index=sc_adata.obs_names, columns=sc_adata.var_names)
sc_df.to_csv(f'{output_dir}/sc_data.tsv', sep='\t')

# 保存单细胞标签
print("[信息] 正在保存 sc_label.tsv...")
# 尝试自动匹配常见的细胞类型列名
cell_type_col = None
for col in ['cell_type', 'celltype', 'CellType', 'cluster', 'leiden', 'louvain']:
    if col in sc_adata.obs.columns:
        cell_type_col = col
        break

if cell_type_col is None:
    print(f"  可用列: {sc_adata.obs.columns.tolist()}")
    cell_type_col = sc_adata.obs.columns[0]  # 默认使用第一列

sc_label = pd.DataFrame({'cell': sc_adata.obs_names, 'cell_type': sc_adata.obs[cell_type_col].values})
sc_label.to_csv(f'{output_dir}/sc_label.tsv', sep='\t', index=False)

# 保存空间表达矩阵
print("[信息] 正在保存 ST_data.tsv...")
st_df = pd.DataFrame(st_adata.X.toarray() if hasattr(st_adata.X, 'toarray') else st_adata.X,
                     index=st_adata.obs_names, columns=st_adata.var_names)
st_df.to_csv(f'{output_dir}/ST_data.tsv', sep='\t')

# 保存坐标信息
print("[信息] 正在保存 coordinates.csv...")
if 'spatial' in st_adata.obsm:
    coords = pd.DataFrame(st_adata.obsm['spatial'], index=st_adata.obs_names, columns=['x', 'y'])
elif st_adata.obs.shape[1] >= 2:
    # 尝试从 obs 中查找 x/y 坐标列
    x_col = None
    y_col = None
    for col in st_adata.obs.columns:
        if 'x' in col.lower():
            x_col = col
        if 'y' in col.lower():
            y_col = col
    if x_col and y_col:
        coords = st_adata.obs[[x_col, y_col]].copy()
        coords.columns = ['x', 'y']
    else:
        coords = st_adata.obs.iloc[:, :2].copy()
        coords.columns = ['x', 'y']
else:
    raise ValueError("未找到空间坐标！")

coords.to_csv(f'{output_dir}/coordinates.csv')

print("[成功] 数据转换完成！")
print(f"文件已保存至: {output_dir}/")
