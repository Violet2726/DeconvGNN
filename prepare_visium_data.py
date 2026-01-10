# 转换 Visium 数据 + 准备单细胞参考数据
import scanpy as sc
import pandas as pd
import numpy as np
import os

print("="*60)
print("准备 Visium 小鼠大脑数据集 + 单细胞参考")
print("="*60)

# 1. 加载 Visium 空间数据
print("\n[Step 1] 加载 Visium 空间数据...")
st_adata = sc.read_h5ad('data/visium_mouse_brain/visium_brain.h5ad')
st_adata.var_names_make_unique()  # 修复重复基因名
print(f"  空间点数量: {st_adata.n_obs}")
print(f"  基因数量: {st_adata.n_vars}")

# 2. 加载单细胞参考数据
print("\n[Step 2] 加载单细胞参考数据...")
sc_adata = sc.read_h5ad('data/starmap/starmap_sc_adata.h5ad')
sc_adata.var_names_make_unique()  # 修复重复基因名
print(f"  细胞数量: {sc_adata.n_obs}")
print(f"  基因数量: {sc_adata.n_vars}")

# 3. 找出共同基因
print("\n[Step 3] 计算共同基因...")
st_genes = set(st_adata.var_names)
sc_genes = set(sc_adata.var_names)
common_genes = st_genes.intersection(sc_genes)
print(f"  Visium 基因数: {len(st_genes)}")
print(f"  单细胞基因数: {len(sc_genes)}")
print(f"  共同基因数: {len(common_genes)}")

if len(common_genes) < 100:
    print("\n⚠️ 警告：共同基因太少！可能是基因名格式不同。")
    print(f"  Visium 示例基因: {list(st_adata.var_names)[:5]}")
    print(f"  单细胞示例基因: {list(sc_adata.var_names)[:5]}")
else:
    print(f"\n✅ 共同基因数量足够 ({len(common_genes)} 个)")

# 4. 创建输出目录
output_dir = 'data/visium_combined'
os.makedirs(output_dir, exist_ok=True)

# 5. 筛选共同基因并保存
print("\n[Step 4] 筛选共同基因并保存...")
common_genes_list = list(common_genes)

# 筛选 Visium 数据
st_filtered = st_adata[:, st_adata.var_names.isin(common_genes_list)].copy()
# 筛选单细胞数据
sc_filtered = sc_adata[:, sc_adata.var_names.isin(common_genes_list)].copy()

print(f"  筛选后 Visium: {st_filtered.shape}")
print(f"  筛选后单细胞: {sc_filtered.shape}")

# 6. 保存为 TSV 格式
print("\n[Step 5] 保存为 TSV 格式...")

# 单细胞表达矩阵
print("  保存 sc_data.tsv...")
sc_df = pd.DataFrame(
    sc_filtered.X.toarray() if hasattr(sc_filtered.X, 'toarray') else sc_filtered.X,
    index=sc_filtered.obs_names,
    columns=sc_filtered.var_names
)
sc_df.to_csv(f'{output_dir}/sc_data.tsv', sep='\t')

# 单细胞标签
print("  保存 sc_label.tsv...")
cell_type_col = None
for col in ['cell_type', 'celltype', 'CellType', 'cluster', 'Cell_class']:
    if col in sc_filtered.obs.columns:
        cell_type_col = col
        break
if cell_type_col is None:
    cell_type_col = sc_filtered.obs.columns[0]
print(f"    使用列 '{cell_type_col}' 作为细胞类型")

sc_label = pd.DataFrame({
    'cell': sc_filtered.obs_names,
    'cell_type': sc_filtered.obs[cell_type_col].values
})
sc_label.to_csv(f'{output_dir}/sc_label.tsv', sep='\t', index=False)

# 空间表达矩阵
print("  保存 ST_data.tsv...")
st_df = pd.DataFrame(
    st_filtered.X.toarray() if hasattr(st_filtered.X, 'toarray') else st_filtered.X,
    index=st_filtered.obs_names,
    columns=st_filtered.var_names
)
st_df.to_csv(f'{output_dir}/ST_data.tsv', sep='\t')

# 空间坐标
print("  保存 coordinates.csv...")
if 'spatial' in st_filtered.obsm:
    coords = pd.DataFrame(
        st_filtered.obsm['spatial'],
        index=st_filtered.obs_names,
        columns=['x', 'y']
    )
else:
    raise ValueError("找不到空间坐标！")
coords.to_csv(f'{output_dir}/coordinates.csv')

print("\n" + "="*60)
print("[SUCCESS] 数据准备完成！")
print("="*60)
print(f"\n文件保存到: {output_dir}/")
print(f"  - sc_data.tsv ({sc_df.shape[0]} 细胞 × {sc_df.shape[1]} 基因)")
print(f"  - sc_label.tsv")
print(f"  - ST_data.tsv ({st_df.shape[0]} 空间点 × {st_df.shape[1]} 基因)")
print(f"  - coordinates.csv ({len(coords)} 个空间点)")
print(f"\n细胞类型: {sc_label['cell_type'].nunique()} 种")
print(sc_label['cell_type'].value_counts())
