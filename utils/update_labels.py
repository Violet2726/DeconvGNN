"""
根据外部元数据（如 Allen Brain 元数据）更新单细胞数据集的细胞类型标签。
用于将粗粒度的标签（如 L2/3 IT）细化为更具体的子类。
"""
import pandas as pd
import gzip
import os

def update_labels(dataset_name):
    print(f"正在更新数据集 {dataset_name} 的标签...")
    combined_dir = f"data/{dataset_name}/combined"
    label_path = os.path.join(combined_dir, "sc_label.tsv")
    backup_path = os.path.join(combined_dir, "sc_label_backup.tsv")
    metadata_path = "data/ref_mouse_cortex_allen/GSE115746_complete_metadata_28706-cells.csv.gz"

    if not os.path.exists(label_path):
        print(f"错误：找不到 {label_path}。")
        return

    # 备份原始标签
    if not os.path.exists(backup_path):
        os.rename(label_path, backup_path)
        print(f"原始标签已备份至 {backup_path}。")
    else:
        # 若备份存在，始终从备份读取以保持状态纯净
        pass

    # 读取当前细胞 (从备份)
    current_labels = pd.read_csv(backup_path, sep='\t')
    
    # 读取元数据
    metadata = pd.read_csv(metadata_path, compression='gzip')
    # 元数据列：使用 sample_name 作为键，映射到 cell_subclass
    metadata_map = metadata.set_index('sample_name')['cell_subclass'].to_dict()

    # 映射标签：若元数据中不存在，则保留原始类型
    def map_detailed(row):
        cell_id = row['cell']
        detailed = metadata_map.get(cell_id)
        if pd.isna(detailed) or detailed == "" or detailed == "No Class":
            return row['cell_type'] # 回退到原始值
        return detailed

    current_labels['cell_type'] = current_labels.apply(map_detailed, axis=1)
    
    # 保存新标签
    current_labels.to_csv(label_path, sep='\t', index=False)
    print(f"更新后的标签已保存至 {label_path}。")
    print("新的细胞类型分布情况：")
    print(current_labels['cell_type'].value_counts().head(15))

if __name__ == "__main__":
    update_labels("V1_Mouse_Brain_Sagittal_Posterior")
    update_labels("V1_Mouse_Brain_Sagittal_Anterior")
    update_labels("V1_Adult_Mouse_Brain_Coronal_Section_1")
    update_labels("CytAssist_11mm_FFPE_Mouse_Embryo")
