"""
整合 Visium 空间数据与 Allen Brain 单细胞数据
为 STdGCN 准备训练数据

输入数据:
1. 单细胞参考: data/ref_mouse_cortex_allen (已有的高质量参考)
2. 空间数据:   data/[DATASET_NAME] (默认 V1_Adult_Mouse_Brain_Coronal_Section_1)

输出数据:
- data/[DATASET_NAME]/combined/
"""
import pandas as pd
import numpy as np
import scanpy as sc
import os
import argparse

# 默认数据集
DEFAULT_DATASET = 'CytAssist_11mm_FFPE_Mouse_Embryo'

def main():
    parser = argparse.ArgumentParser(description='整合 Visium 和 Allen Brain 数据')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASET,
                        help=f'Visium 数据集目录名称 (默认: {DEFAULT_DATASET})')
    args = parser.parse_args()

    # ============================================================
    # 1. 配置路径
    # ============================================================
    st_source = args.dataset
    
    # 输入目录 - 固定使用现有的高质量参考目录
    sc_dir = "data/ref_mouse_cortex_allen"  
    st_dir = f"data/{st_source}"
    
    # 输出目录 (直接在空间数据目录下创建 combined 子目录)
    output_dir = f"data/{st_source}/combined"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("生成 STdGCN 训练数据")
    print("=" * 60)
    print(f"数据源:")
    print(f"  - 单细胞参考: {sc_dir}")
    print(f"  - 空间数据:   {st_dir}")
    print(f"输出目标:")
    print(f"  - {output_dir}")
    
    # ============================================================
    # 2. 加载单细胞数据 (Allen Brain Reference)
    # ============================================================
    print("\n" + "-" * 60)
    print("[Step 1] 加载 Allen Brain 单细胞参考")
    print("-" * 60)
    
    sc_data_path = os.path.join(sc_dir, "sc_data.tsv")
    sc_label_path = os.path.join(sc_dir, "sc_label.tsv")
    
    if not os.path.exists(sc_data_path) or not os.path.exists(sc_label_path):
        print(f"[ERROR] 单细胞参考数据缺失!")
        print(f"  检查路径: {sc_dir}")
        print("  请确保目录中包含 'sc_data.tsv' 和 'sc_label.tsv'")
        return

    print("  读取 sc_data.tsv (可能需要几分钟)...")
    sc_data = pd.read_csv(sc_data_path, sep='\t', index_col=0)
    print(f"  ✓ 形状: {sc_data.shape}")
    
    print("  读取 sc_label.tsv...")
    sc_label = pd.read_csv(sc_label_path, sep='\t')
    print(f"  ✓ 细胞类型数: {sc_label['cell_type'].nunique()}")
    print(f"  ✓ 细胞类型: {sc_label['cell_type'].unique()}")
    
    sc_genes = set(sc_data.columns)

    # ============================================================
    # 3. 加载空间数据 (Visium)
    # ============================================================
    print("\n" + "-" * 60)
    print("[Step 2] 加载 Visium 空间数据")
    print("-" * 60)
    
    st_data_path = os.path.join(st_dir, "ST_data.tsv")
    st_coords_path = os.path.join(st_dir, "coordinates.csv")

    # 如果 TSV 不存在，尝试从 h5ad 加载 (处理旧数据格式情况)
    if not os.path.exists(st_data_path):
        print("  [WARN] 未找到 ST_data.tsv，尝试搜索 h5ad 文件...")
        h5ad_path = None
        if os.path.exists(st_dir):
            for fname in os.listdir(st_dir):
                if fname.endswith('.h5ad'):
                    h5ad_path = os.path.join(st_dir, fname)
                    break
        
        if h5ad_path:
            print(f"  从 h5ad 加载: {h5ad_path}")
            st_adata = sc.read_h5ad(h5ad_path)
            st_adata.var_names_make_unique()
            
            # 提取并保存 TSV，方便后续加载
            if hasattr(st_adata.X, 'toarray'):
                st_data = pd.DataFrame(st_adata.X.toarray(), index=st_adata.obs_names, columns=st_adata.var_names)
            else:
                st_data = pd.DataFrame(st_adata.X, index=st_adata.obs_names, columns=st_adata.var_names)
            
            # 提取坐标
            st_coords = pd.DataFrame(st_adata.obsm['spatial'], index=st_adata.obs_names, columns=['x', 'y'])
        else:
            print(f"[ERROR] 无法找到空间数据。请确保 data/{st_source} 下包含 ST_data.tsv 或 .h5ad")
            print(f"  提示: 可以使用 'python utils/download_visium_data.py' 重新下载")
            return
    else:
        print("  读取 ST_data.tsv...")
        st_data = pd.read_csv(st_data_path, sep='\t', index_col=0)
        st_coords = pd.read_csv(st_coords_path, index_col=0)

    print(f"  ✓ 空间点数: {st_data.shape[0]}")
    print(f"  ✓ 基因数:   {st_data.shape[1]}")
    st_genes = set(st_data.columns)

    # ============================================================
    # 4. 计算并筛选共同基因
    # ============================================================
    print("\n" + "-" * 60)
    print("[Step 3] 整合数据 (基因求交集)")
    print("-" * 60)
    
    common_genes = sc_genes.intersection(st_genes)
    print(f"  单细胞基因: {len(sc_genes)}")
    print(f"  空间基因:   {len(st_genes)}")
    print(f"  ✓ 共同基因: {len(common_genes)}")
    
    # 自动大小写修复逻辑
    if len(common_genes) < 500:
        print("\n  ⚠️ 警告: 共同基因太少，尝试忽略大小写匹配...")
        sc_upper = {g.upper(): g for g in sc_data.columns}
        st_upper = {g.upper(): g for g in st_data.columns}
        
        upper_common = set(sc_upper.keys()).intersection(set(st_upper.keys()))
        if len(upper_common) > 500:
            print(f"  ✓ 忽略大小写后找到 {len(upper_common)} 个共同基因，正在修复...")
            
            # 映射回原始列名 (以单细胞为准)
            sc_keep_cols = [sc_upper[g] for g in upper_common]
            st_rename_map = {st_upper[g]: sc_upper[g] for g in upper_common}
            
            # 重命名空间数据的列以匹配单细胞
            st_keep_cols = []
            st_final_rename = {}
            for g_up in upper_common:
                orig_st = st_upper[g_up]
                st_keep_cols.append(orig_st)
                st_final_rename[orig_st] = st_rename_map[orig_st]
            
            sc_data = sc_data[sc_keep_cols]
            st_data = st_data[st_keep_cols].rename(columns=st_final_rename)
            
            common_genes_list = sorted(list(sc_data.columns))
            print("  ✓ 修复完成")
        else:
            print("  ❌ 修复失败，请检查数据是否匹配 (不同物种?)。")
            return
    else:
        common_genes_list = sorted(list(common_genes))
        sc_data = sc_data[common_genes_list]
        st_data = st_data[common_genes_list]

    print(f"  整合后维度: {len(common_genes_list)} Genes")

    # ============================================================
    # 5. 保存结果
    # ============================================================
    print("\n" + "-" * 60)
    print("[Step 4] 保存到 combined 目录")
    print("-" * 60)
    
    # 保存 sc_data.tsv
    print("  保存 sc_data.tsv...")
    sc_data.to_csv(os.path.join(output_dir, "sc_data.tsv"), sep='\t')
    
    # 保存 sc_label.tsv
    print("  保存 sc_label.tsv...")
    sc_label.to_csv(os.path.join(output_dir, "sc_label.tsv"), sep='\t', index=False)
    
    # 保存 ST_data.tsv
    print("  保存 ST_data.tsv...")
    st_data.to_csv(os.path.join(output_dir, "ST_data.tsv"), sep='\t')
    
    # 保存 coordinates.csv
    print("  保存 coordinates.csv...")
    st_coords.to_csv(os.path.join(output_dir, "coordinates.csv"))
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 完成！")
    print(f"数据已保存在: {output_dir}")
    print("=" * 60)
    print(f"后续操作提醒: 训练脚本 Tutorial.py 应指向上述目录。")

if __name__ == "__main__":
    main()
