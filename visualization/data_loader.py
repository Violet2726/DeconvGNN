
import os
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List, Dict

# 数据目录配置 (初始为空)
DATA_DIRS: Dict[str, str] = {}

@st.cache_data
def load_results(result_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    加载反卷积结果和坐标文件。
    自动在结果目录、父目录及 combined 子目录中搜索 coordinates.csv。

    Returns:
        (predict_df, coords): 预测结果与坐标数据 DataFrame。若加载失败返回 (None, None)。
    """
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None
    
    predict_df = pd.read_csv(predict_path, index_col=0)
    
    # 尝试加载坐标
    coords = None
    
    # 优先检查结果目录本身是否包含坐标文件
    coord_in_result = os.path.join(result_dir, "coordinates.csv")
    # 检查父目录（数据集根目录）
    parent_dir = os.path.dirname(result_dir)
    coord_in_parent = os.path.join(parent_dir, "coordinates.csv")
    # 检查父目录下的 combined 子目录
    coord_in_combined = os.path.join(parent_dir, "combined", "coordinates.csv")
    
    # 搜索顺序：结果目录 -> 父目录 -> combined目录
    search_paths = [coord_in_result, coord_in_parent, coord_in_combined]
    
    for coord_path in search_paths:
        if os.path.exists(coord_path):
            try:
                coords = pd.read_csv(coord_path, index_col=0)
                if len(coords) == len(predict_df):
                    break
            except Exception:
                continue
    
    return predict_df, coords

def get_cell_types(predict_df: pd.DataFrame) -> List[str]:
    """提取预测结果中的细胞类型列表。"""
    return predict_df.columns.tolist()
