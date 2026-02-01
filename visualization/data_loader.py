
import os
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List, Dict

# 数据目录配置 (初始为空)
DATA_DIRS: Dict[str, str] = {}

# ========== 缓存配置 ==========
CACHE_TTL = 3600  # 缓存有效期 1 小时
CACHE_MAX_ENTRIES = 10  # 最多缓存 10 个数据集

@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
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
    
    print(f"[DataLoader] Loading predict_df from: {predict_path}")
    print(f"[DataLoader] Searching for coords in: {search_paths}")
    
    for coord_path in search_paths:
        if os.path.exists(coord_path):
            try:
                temp_coords = pd.read_csv(coord_path, index_col=0)
                print(f"[DataLoader] Found {coord_path}, rows: {len(temp_coords)} vs {len(predict_df)}")
                
                # Loose check: verify if indices overlap significantly instead of strict length match
                # strict checking often fails due to slight filtering differences
                common_indices = predict_df.index.intersection(temp_coords.index)
                match_ratio = len(common_indices) / len(predict_df)
                
                if match_ratio > 0.9: # 90% overlap allowed
                     coords = temp_coords.loc[predict_df.index] # Realign
                     print(f"[DataLoader] Valid match found! Aligned {len(coords)} rows.")
                     break
                else:
                    print(f"[DataLoader] Mismatch indices. Overlap ratio: {match_ratio:.2f}")

            except Exception as e:
                print(f"[DataLoader] Error reading {coord_path}: {e}")
                continue
    
    if coords is None:
        print("[DataLoader] FAILED to find matching coordinates file.")
        
    return predict_df, coords

def get_cell_types(predict_df: pd.DataFrame) -> List[str]:
    """提取预测结果中的细胞类型列表。"""
    return predict_df.columns.tolist()
