# -*- coding: utf-8 -*-
"""
根据 Allen Brain 元数据细化单细胞标签。

训练数据中的 `sc_label.tsv` 可能只包含较粗粒度的细胞类型。该脚本使用
Allen Brain 的完整元数据，把标签映射到 `cell_subclass`，从而让 STdGCN
学习更细的细胞亚群组成。
"""
import pandas as pd
import gzip
import os

DATASETS = [
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Adult_Mouse_Brain_Coronal_Section_1",
    "CytAssist_11mm_FFPE_Mouse_Embryo"
]

def update_labels(dataset_name):
    """
    更新指定数据集 `combined/sc_label.tsv` 中的细胞类型标签。

    参数:
        dataset_name: `data/` 下的数据集目录名。

    返回:
        None: 更新后的标签写回 `sc_label.tsv`，原始文件会保存在备份中。
    """
    print(f"正在更新数据集 {dataset_name} 的标签...")
    combined_dir = f"data/{dataset_name}/combined"
    label_path = os.path.join(combined_dir, "sc_label.tsv")
    backup_path = os.path.join(combined_dir, "sc_label_backup.tsv")
    metadata_path = "data/ref_mouse_cortex_allen/GSE115746_complete_metadata_28706-cells.csv.gz"

    if not os.path.exists(label_path):
        print(f"错误：找不到 {label_path}。")
        return

    # 首次运行时备份原始标签；重复运行时始终从备份读，避免多次映射叠加。
    if not os.path.exists(backup_path):
        os.rename(label_path, backup_path)
        print(f"原始标签已备份至 {backup_path}。")
    else:
        # 若备份存在，始终从备份读取以保持状态纯净。
        pass

    # 读取当前细胞标签（从备份恢复原始状态）。
    current_labels = pd.read_csv(backup_path, sep='\t')
    
    # 读取 Allen Brain 元数据。
    metadata = pd.read_csv(metadata_path, compression='gzip')
    # 元数据列：使用 sample_name 作为键，映射到 cell_subclass。
    metadata_map = metadata.set_index('sample_name')['cell_subclass'].to_dict()

    # 映射标签：若元数据中不存在，则保留原始类型，避免无匹配细胞丢失标签。
    def map_detailed(row):
        """把单个细胞标签映射到更细粒度的 subclass。"""
        cell_id = row['cell']
        detailed = metadata_map.get(cell_id)
        if pd.isna(detailed) or detailed == "" or detailed == "No Class":
            return row['cell_type']  # 回退到原始值。
        return detailed

    current_labels['cell_type'] = current_labels.apply(map_detailed, axis=1)
    
    # 保存新标签，并打印前 15 个高频类型用于人工快速检查。
    current_labels.to_csv(label_path, sep='\t', index=False)
    print(f"更新后的标签已保存至 {label_path}。")
    print("新的细胞类型分布情况：")
    print(current_labels['cell_type'].value_counts().head(15))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='更新单细胞标签')
    parser.add_argument('--dataset', type=str, nargs='+', 
                        default=DATASETS,
                        help='要更新的数据集名称 (支持多个, 空格分隔)')
    args = parser.parse_args()
    
    for ds in args.dataset:
        update_labels(ds)
