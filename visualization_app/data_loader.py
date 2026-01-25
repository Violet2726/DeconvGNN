
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

    Args:
        result_dir (str): 结果数据的根目录路径。

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: 
            - predict_df: 预测结果 DataFrame (index=位置, columns=细胞类型)
            - coords: 坐标信息 DataFrame (index=位置, columns=['x', 'y'])
    """
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None
    
    predict_df = pd.read_csv(predict_path, index_col=0)
    
    # 尝试加载坐标
    coords = None
    
    # 优先检查结果目录本身是否包含坐标文件
    coord_in_result = os.path.join(result_dir, "coordinates.csv")
    # 再检查与结果目录同级的 data 目录
    parent_dir = os.path.dirname(result_dir)
    coord_in_parent_data = os.path.join(parent_dir, "data", "coordinates.csv")
    
    # 搜索顺序：结果目录 -> 父目录/data -> 预设目录
    search_paths = [coord_in_result, coord_in_parent_data, 
                    "data/visium_combined/coordinates.csv", 
                    "data/seqfish_tsv/coordinates.csv", 
                    "data/starmap_tsv/coordinates.csv"]
    
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
    """
    获取预测结果中的细胞类型列表。

    Args:
        predict_df (pd.DataFrame): 预测结果数据框。

    Returns:
        List[str]: 细胞类型名称列表。
    """
    return predict_df.columns.tolist()
