"""
用于在不重新训练模型的情况下，根据已有的预测数据重新生成交互式和静态图表。
主要用于调整可视化的样式或补充缺失的图表。
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到路径以导入模块
sys.path.append(os.getcwd())

from visualization.utils import generate_and_save_interactive_assets

DATASETS = [
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Adult_Mouse_Brain_Coronal_Section_1",
    "CytAssist_11mm_FFPE_Mouse_Embryo"
]

def generate_plot(dataset_name):
    print(f"正在为 {dataset_name} 重新生成图表...")
    
    results_dir = f"./data/{dataset_name}/results"
    predict_path = os.path.join(results_dir, "predict_result.csv")
    coor_path = f"./data/{dataset_name}/combined/coordinates.csv"
    
    if not os.path.exists(predict_path):
        print(f"错误：未找到 {predict_path}。请确认是否已完成训练。")
        return

    # 加载预测结果
    predict_df = pd.read_csv(predict_path, index_col=0)
    cell_types = predict_df.columns.tolist()
    
    # 加载坐标数据 (格式: Barcode, x, y)
    coor_df = pd.read_csv(coor_path, header=0, index_col=0)
    
    # 对齐坐标与预测结果
    # 仅保留共有的斑点
    common_indices = predict_df.index.intersection(coor_df.index)
    
    if len(common_indices) != len(predict_df):
        print(f"警告：Barcode 不匹配。预测: {len(predict_df)}, 坐标: {len(coor_df)}, 交集: {len(common_indices)}")
    
    predict_df = predict_df.loc[common_indices]
    coor_df = coor_df.loc[common_indices]
    
    
    try:
        generate_and_save_interactive_assets(predict_df, coor_df, results_dir)
        print("成功！高分辨率图表已生成。")
    except Exception as e:
        print(f"生成图表时出错: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='在不重新训练的情况下重新生成交互式图表')
    parser.add_argument('--dataset', type=str, nargs='+', default=DATASETS,
                        help='数据集名称，使用空格分隔多个 (例如: --dataset Data1 Data2)')
    args = parser.parse_args()
    
    datasets = args.dataset
    
    for i, ds in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] 正在处理数据集: {ds}")
        generate_plot(ds)
