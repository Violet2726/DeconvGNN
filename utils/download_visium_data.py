"""
整合的 Visium 数据集下载工具
支持下载 10x Genomics 官方示例数据集并格式化为 STdGCN 格式
"""
import scanpy as sc
import pandas as pd
import os
import argparse

# 可用的数据集列表 (Scanpy 支持的 sample_id)
DATASETS = {
    'coronal': 'V1_Adult_Mouse_Brain_Coronal_Section_1',
    'sagittal_anterior': 'V1_Mouse_Brain_Sagittal_Anterior',
    'sagittal_posterior': 'V1_Mouse_Brain_Sagittal_Posterior',
}

def download_and_format_visium(dataset_key='coronal', output_root='data'):
    """
    下载并格式化 Visium 数据
    :param dataset_key: 数据集简称 (key in DATASETS)
    :param output_root: 根数据目录
    """
    if dataset_key not in DATASETS:
        print(f"[ERROR] 未知的数据集。可用选项: {list(DATASETS.keys())}")
        return

    sample_id = DATASETS[dataset_key]
    print(f"=" * 60)
    print(f"[INFO] 正在处理数据集: {dataset_key} (Sample ID: {sample_id})")
    print(f"=" * 60)

    # 目标目录: data/V1_Adult_Mouse_Brain_Coronal_Section_1
    dataset_dir = os.path.join(output_root, sample_id)
    
    # 检查是否已存在
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'ST_data.tsv')):
        print(f"[INFO] 目录 {dataset_dir} 已存在且包含 ST_data.tsv，跳过下载。")
        return dataset_dir

    try:
        print(f"[INFO] 正在从 10x/Scanpy 下载数据...")
        # 1. 下载数据 (包含 hires 图像)
        adata = sc.datasets.visium_sge(sample_id=sample_id, include_hires_tiff=True)
        
        # 2. 创建输出目录
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 3. 生成 ST_data.tsv (表达矩阵)
        print(f"[INFO] 生成 ST_data.tsv...")
        if hasattr(adata.X, "toarray"):
            df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
        else:
            df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        
        st_path = os.path.join(dataset_dir, "ST_data.tsv")
        df.to_csv(st_path, sep='\t')
        print(f"  ✓ 保存成功: {df.shape}")
        
        # 4. 生成 coordinates.csv (空间坐标)
        print(f"[INFO] 生成 coordinates.csv...")
        if 'spatial' in adata.obsm:
            coords = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
            coords.index.name = 'Barcode'
            coords_path = os.path.join(dataset_dir, "coordinates.csv")
            coords.to_csv(coords_path)
            print(f"  ✓ 保存成功: {coords.shape}")
        else:
            print(f"  ⚠️ Warning: 未找到 spatial 坐标信息!")

        # 5. 可选: 保存为 h5ad 备份
        h5ad_path = os.path.join(dataset_dir, "filtered_feature_bc_matrix.h5ad")
        adata.write_h5ad(h5ad_path)
        print(f"  ✓ h5ad 备份已保存")

        print(f"\n[SUCCESS] 全部完成！数据位于: {dataset_dir}")
        print(f"注意: 此目录仅包含空间数据(ST)。如需训练，请记得在该目录下创建 'combined' 子目录并整合单细胞参考数据。")
        
    except Exception as e:
        print(f"[ERROR] 下载或处理失败: {e}")
        # 如果下载一半失败，可能需要清理缓存
        print("提示: 如果是因为网络问题，请检查网络连接后重试。")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='下载 10x Visium 空间转录组数据')
    parser.add_argument('--dataset', type=str, default='coronal', 
                        choices=DATASETS.keys(), help='要下载的数据集名称')
    args = parser.parse_args()

    download_and_format_visium(args.dataset)
