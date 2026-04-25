# -*- coding: utf-8 -*-
"""
根据已有预测结果重新生成可视化资源。

该脚本不会重新训练模型，只读取 `predict_result.csv` 和坐标文件，重新生成
Streamlit 可视化页面使用的交互式背景图资源。适用于修改图表样式或补齐
历史结果目录中缺失的可视化文件。
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys

# 添加项目根目录到 import 路径，便于从任意位置执行脚本。
sys.path.append(os.getcwd())

from visualization.utils import generate_and_save_interactive_assets

DATASETS = [
    "V1_Mouse_Brain_Sagittal_Posterior",
    "V1_Mouse_Brain_Sagittal_Anterior",
    "V1_Adult_Mouse_Brain_Coronal_Section_1",
    "CytAssist_11mm_FFPE_Mouse_Embryo"
]

def generate_plot(dataset_name):
    """
    为单个数据集重新生成可视化资源。

    参数:
        dataset_name: `data/` 下的数据集目录名。

    返回:
        None: 图表资源直接写入该数据集的 `results/` 目录。
    """
    print(f"正在为 {dataset_name} 重新生成图表...")
    
    results_dir = f"./data/{dataset_name}/results"
    predict_path = os.path.join(results_dir, "predict_result.csv")
    coor_path = f"./data/{dataset_name}/combined/coordinates.csv"
    
    if not os.path.exists(predict_path):
        print(f"错误：未找到 {predict_path}。请确认是否已完成训练。")
        return

    # 1. 加载模型预测结果。列为细胞类型，行为真实空间斑点。
    predict_df = pd.read_csv(predict_path, index_col=0)
    cell_types = predict_df.columns.tolist()
    
    # 2. 加载坐标数据，格式为 `Barcode, x, y`。
    coor_df = pd.read_csv(coor_path, header=0, index_col=0)
    
    # 3. 对齐坐标与预测结果，仅保留二者共有的斑点。
    #    这样可以避免训练或转换过程中缺失 barcode 导致的绘图错位。
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
