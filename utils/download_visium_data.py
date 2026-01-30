"""
扩展的 Visium 数据集下载工具
支持下载 10x Genomics 官方示例数据集 (包括 11mm 大视野数据集)
"""
import scanpy as sc
import pandas as pd
import os
import argparse
import subprocess
import shutil

# 可用的数据集列表 (Scanpy 支持的 sample_id)
# 这些是 scanpy.datasets.visium_sge 内置支持的
DATASETS = {
    # 'coronal': 'V1_Adult_Mouse_Brain_Coronal_Section_1',
    # 'sagittal_anterior': 'V1_Mouse_Brain_Sagittal_Anterior',
    # 'sagittal_posterior': 'V1_Mouse_Brain_Sagittal_Posterior',
    # 新增: 11mm 大视野数据集 (斑点数 > 5000)
    # 注：Scanpy 内置函数可能不支持直接下载该ID，我们需要自定义下载逻辑
    'mouse_embryo_11mm': 'CytAssist_11mm_FFPE_Mouse_Embryo', 
}

# 10x 官方下载链接映射 (针对 Scanpy 不直接支持的数据集)
EXTERNAL_URLS = {
    'mouse_embryo_11mm': {
        'h5': 'https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_11mm_FFPE_Mouse_Embryo/CytAssist_11mm_FFPE_Mouse_Embryo_filtered_feature_bc_matrix.h5',
        'image': 'https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_11mm_FFPE_Mouse_Embryo/CytAssist_11mm_FFPE_Mouse_Embryo_spatial.tar.gz'
    }
}

def download_file(url, target_path):
    """使用 curl 下载文件"""
    if os.path.exists(target_path):
        print(f"[INFO] 文件已存在: {target_path}")
        return True
    
    try:
        cmd = ['curl', '-o', target_path, url]
        print(f"[EXEC] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 下载失败: {e}")
        return False

def process_external_dataset(dataset_key, output_root):
    """处理不在 Scanpy 内置列表中的外部数据集"""
    sample_id = DATASETS[dataset_key]
    urls = EXTERNAL_URLS[dataset_key]
    
    # 目标目录结构: output_root/sample_id
    dataset_dir = os.path.join(output_root, sample_id)
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"[INFO] 正在手动下载数据集: {dataset_key} (ID: {sample_id})")
    
    # 1. 下载表达矩阵 (H5)
    h5_path = os.path.join(dataset_dir, "filtered_feature_bc_matrix.h5")
    if not download_file(urls['h5'], h5_path):
        return
        
    # 2. 下载并解压空间图像数据
    spatial_tar_path = os.path.join(dataset_dir, "spatial.tar.gz")
    if download_file(urls['image'], spatial_tar_path):
        print(f"[INFO] 解压空间数据...")
        # 解压到 dataset_dir (10x 的 tar 包通常包含 spatial/ 文件夹)
        subprocess.run(['tar', '-xzvf', spatial_tar_path, '-C', dataset_dir], check=True)
        # 清理压缩包
        os.remove(spatial_tar_path)
    
    # 3. 兼容性处理: Scanpy 可能寻找 tissue_positions_list.csv (v1) 而非 tissue_positions.csv (v2/CytAssist)
    spatial_dir = os.path.join(dataset_dir, "spatial")
    v2_pos = os.path.join(spatial_dir, "tissue_positions.csv")
    v1_pos = os.path.join(spatial_dir, "tissue_positions_list.csv")
    if os.path.exists(v2_pos) and not os.path.exists(v1_pos):
        print(f"[INFO] 创建坐标文件符号链接以兼容 Scanpy...")
        os.symlink("tissue_positions.csv", v1_pos)

    # 4. 使用 Scanpy 读取并转换格式
    print(f"[INFO] 正在转换数据格式...")
    adata = sc.read_visium(path=dataset_dir)
    adata.var_names_make_unique()
    
    # 5. 生成 ST_data.tsv
    print(f"[INFO] 生成 ST_data.tsv...")
    if hasattr(adata.X, "toarray"):
        df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
    else:
        df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    
    st_path = os.path.join(dataset_dir, "ST_data.tsv")
    df.to_csv(st_path, sep='\t')
    print(f"  ✓ 保存成功: {df.shape} (Spots: {df.shape[0]}, Genes: {df.shape[1]})")
    
    # 5. 生成 coordinates.csv
    print(f"[INFO] 生成 coordinates.csv...")
    if 'spatial' in adata.obsm:
        # Visium 的 spatial 坐标通常在 adata.obsm['spatial']
        # 注意：需要确认是像素坐标还是物理坐标，通常 Scanpy 读取的是像素坐标
        coords = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
        coords.index.name = 'Barcode'
        coords_path = os.path.join(dataset_dir, "coordinates.csv")
        coords.to_csv(coords_path)
        print(f"  ✓ 保存成功: {coords.shape}")
    
    # 6. 保存 h5ad 备份
    h5ad_path = os.path.join(dataset_dir, "filtered_feature_bc_matrix.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"  ✓ h5ad 备份已保存")
    
    print(f"\n[SUCCESS] 全部完成！数据位于: {dataset_dir}")

def download_and_format_visium(dataset_key='coronal', output_root='data'):
    """主下载函数"""
    if dataset_key not in DATASETS:
        print(f"[ERROR] 未知的数据集。可用选项: {list(DATASETS.keys())}")
        return

    # 检查是否为外部数据集 (需要手动下载)
    if dataset_key in EXTERNAL_URLS:
        process_external_dataset(dataset_key, output_root)
        return

    # 处理 Scanpy 内置数据集
    sample_id = DATASETS[dataset_key]
    dataset_dir = os.path.join(output_root, sample_id)
    
    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, 'ST_data.tsv')):
        print(f"[INFO] 目录 {dataset_dir} 已存在且包含 ST_data.tsv，跳过下载。")
        return

    try:
        print(f"[INFO] 正在从 10x/Scanpy 下载数据: {sample_id}")
        adata = sc.datasets.visium_sge(sample_id=sample_id, include_hires_tiff=True)
        
        os.makedirs(dataset_dir, exist_ok=True)
        
        # 生成 ST_data.tsv
        print(f"[INFO] 生成 ST_data.tsv...")
        if hasattr(adata.X, "toarray"):
            df = pd.DataFrame(adata.X.toarray(), index=adata.obs_names, columns=adata.var_names)
        else:
            df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
        
        st_path = os.path.join(dataset_dir, "ST_data.tsv")
        df.to_csv(st_path, sep='\t')
        print(f"  ✓ 保存成功: {df.shape}")
        
        # 生成 coordinates.csv
        print(f"[INFO] 生成 coordinates.csv...")
        if 'spatial' in adata.obsm:
            coords = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
            coords.index.name = 'Barcode'
            coords_path = os.path.join(dataset_dir, "coordinates.csv")
            coords.to_csv(coords_path)
            print(f"  ✓ 保存成功: {coords.shape}")

        # 保存 h5ad
        h5ad_path = os.path.join(dataset_dir, "filtered_feature_bc_matrix.h5ad")
        adata.write_h5ad(h5ad_path)

        # 准备 STdGCN 训练所需的 combined 目录
        combined_dir = os.path.join(dataset_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        shutil.copy(st_path, os.path.join(combined_dir, "ST_data.tsv"))
        shutil.copy(coords_path, os.path.join(combined_dir, "coordinates.csv"))

        print(f"\n[SUCCESS] 全部完成！数据位于: {dataset_dir}")
        print(f"[INFO] 训练所需文件已同步至: {combined_dir}")
        
    except Exception as e:
        print(f"[ERROR] 下载或处理失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='下载 10x Visium 空间转录组数据')
    parser.add_argument('--dataset', type=str, default='mouse_embryo_11mm', 
                        choices=DATASETS.keys(), help='要下载的数据集名称')
    args = parser.parse_args()

    download_and_format_visium(args.dataset)
