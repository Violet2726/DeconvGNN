import os
import requests
import tarfile
import h5py
import pandas as pd
import scipy.sparse
import numpy as np
import shutil

# 配置
DATASET_NAME = 'CytAssist_11mm_FFPE_Mouse_Embryo'
SAMPLE_ID = DATASET_NAME
OUTPUT_ROOT = 'data'
URL_H5 = 'https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_11mm_FFPE_Mouse_Embryo/CytAssist_11mm_FFPE_Mouse_Embryo_filtered_feature_bc_matrix.h5'
URL_SPATIAL = 'https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_11mm_FFPE_Mouse_Embryo/CytAssist_11mm_FFPE_Mouse_Embryo_spatial.tar.gz'

def download_file(url, target_path):
    if os.path.exists(target_path):
        print(f"文件已存在: {target_path}")
        return
    print(f"正在下载 {url} ...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(target_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("下载完成")

def read_10x_h5(h5_file):
    print(f"读取 H5 文件: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        # 10x H5 版本差异处理
        if 'matrix' in f:
            group = f['matrix']
        else:
            group = f
            
        M, N = group['shape'][()]
        data = group['data'][()]
        indices = group['indices'][()]
        indptr = group['indptr'][()]
        
        # 基因名和Barcode
        features = group['features']['name'][()].astype(str)
        barcodes = group['barcodes'][()].astype(str)
        
        # 构建稀疏矩阵
        mat = scipy.sparse.csr_matrix((data, indices, indptr), shape=(N, M)) # N cells, M genes
        
        df = pd.DataFrame.sparse.from_spmatrix(mat, index=barcodes, columns=features)
        # 转换为密集型如果内存允许 (这里为了后续处理方便，且STdGCN通常需要密集输入)
        # 注意: 6000x20000 float32 约 480MB，可以接受
        return df.sparse.to_dense()

def main():
    dataset_dir = os.path.join(OUTPUT_ROOT, SAMPLE_ID)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 1. 下载文件
    h5_path = os.path.join(dataset_dir, "filtered_feature_bc_matrix.h5")
    spatial_tar_path = os.path.join(dataset_dir, "spatial.tar.gz")
    
    download_file(URL_H5, h5_path)
    download_file(URL_SPATIAL, spatial_tar_path)
    
    # 2. 解压 Spatial
    print("解压 spatial 数据...")
    with tarfile.open(spatial_tar_path) as tar:
        # 安全解压
        def is_within_directory(directory, target):
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
            prefix = os.path.commonprefix([abs_directory, abs_target])
            return prefix == abs_directory
        
        tar.extractall(dataset_dir)
        
    # 3. 处理 ST_data.tsv
    try:
        df = read_10x_h5(h5_path)
        st_path = os.path.join(dataset_dir, "ST_data.tsv")
        df.to_csv(st_path, sep='\t')
        print(f"ST_data.tsv 保存成功: {df.shape}")
    except Exception as e:
        print(f"处理 H5 失败: {e}")
        return

    # 4. 处理 coordinates.csv
    spatial_dir = os.path.join(dataset_dir, "spatial")
    # 尝试不同的文件名 (V1 vs V2)
    pos_file = os.path.join(spatial_dir, "tissue_positions.csv")
    if not os.path.exists(pos_file):
        pos_file = os.path.join(spatial_dir, "tissue_positions_list.csv")
    
    if os.path.exists(pos_file):
        print(f"读取坐标文件: {pos_file}")
        # 检测是否有 header
        with open(pos_file, 'r') as f:
            first_line = f.readline()
            has_header = "in_tissue" in first_line or "barcode" in first_line
            
        if has_header:
            coords_all = pd.read_csv(pos_file)
        else:
            coords_all = pd.read_csv(pos_file, header=None, names=[
                "barcode", "in_tissue", "array_row", "array_col", 
                "pxl_row_in_fullres", "pxl_col_in_fullres"
            ])
            
        # 过滤 in_tissue == 1
        coords_tissue = coords_all[coords_all["in_tissue"] == 1].copy()
        
        # 提取 needed columns
        # 注意: columns 必须是对齐的. STdGCN 期望: index=barcode, columns=['x', 'y']
        coords_out = pd.DataFrame()
        coords_out.index = coords_tissue["barcode"]
        # px_col -> x, px_row -> y (通常图像坐标系)
        # 或者是 array_col/row? 通常 visualization 用 pixel 坐标
        coords_out['x'] = coords_tissue["pxl_col_in_fullres"].values
        coords_out['y'] = coords_tissue["pxl_row_in_fullres"].values
        coords_out.index.name = "Barcode"
        
        # 确保只保留在表达矩阵中的点
        valid_barcodes = df.index.intersection(coords_out.index)
        coords_out = coords_out.loc[valid_barcodes]
        
        coords_path = os.path.join(dataset_dir, "coordinates.csv")
        coords_out.to_csv(coords_path)
        print(f"coordinates.csv 保存成功: {coords_out.shape}")
    else:
        print("未找到 tissue_positions 文件!")

    # 5. 为了让 STdGCN 运行，创建 'combined' 目录并将文件移入
    # Tutorial.py 中 paths = '.../combined'
    combined_dir = os.path.join(dataset_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # 复制文件
    shutil.copy(st_path, os.path.join(combined_dir, "ST_data.tsv"))
    if os.path.exists(coords_path):
        shutil.copy(coords_path, os.path.join(combined_dir, "coordinates.csv"))
        
    print(f"所有数据已准备就绪: {combined_dir}")

if __name__ == "__main__":
    main()
