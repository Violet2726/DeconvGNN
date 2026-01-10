# 下载 10x Visium 小鼠大脑数据集
import scanpy as sc
import pandas as pd
import os

print("[INFO] 下载 10x Visium 小鼠大脑数据集...")
print("这是一个公开的示例数据集，空间点数量适中。\n")

# 使用 scanpy 内置的 visium 数据集下载功能
# 这会自动下载一个小鼠大脑的 Visium 数据
adata = sc.datasets.visium_sge(sample_id="V1_Mouse_Brain_Sagittal_Anterior")

print(f"[SUCCESS] 下载完成！")
print(f"  空间点数量: {adata.n_obs}")
print(f"  基因数量: {adata.n_vars}")
print(f"  形状: {adata.shape}")

# 查看空间坐标
print(f"\n空间坐标信息:")
print(f"  obsm keys: {list(adata.obsm.keys())}")

# 创建输出目录
output_dir = 'data/visium_mouse_brain'
os.makedirs(output_dir, exist_ok=True)

# 保存为 h5ad
adata.write_h5ad(f'{output_dir}/visium_brain.h5ad')
print(f"\n[INFO] 数据已保存到: {output_dir}/visium_brain.h5ad")

# 显示一些基本信息
print(f"\n数据预览:")
print(adata)
