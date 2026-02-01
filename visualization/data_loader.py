
import os
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List, Dict

# 数据目录配置 (初始为空)
DATA_DIRS: Dict[str, str] = {}

# --- 运行时 IO 模型配置 ---
CACHE_TTL = 3600          # 内存持久化时长：1 小时
CACHE_MAX_ENTRIES = 10     # 最大驻留数据集实例数量

@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def load_results(result_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    数据集加载核心引擎。
    从物理磁盘读取反卷积结果，并执行分级探测策略搜索关联的空间位点坐标文件。
    """
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None
    
    predict_df = pd.read_csv(predict_path, index_col=0)
    
    # 尝试加载坐标
    coords = None
    
    # 执行分级路径探测机制：结果目录 -> 数据集父目录 -> combined 合并目录
    coord_in_result = os.path.join(result_dir, "coordinates.csv")
    parent_dir = os.path.dirname(result_dir)
    coord_in_parent = os.path.join(parent_dir, "coordinates.csv")
    coord_in_combined = os.path.join(parent_dir, "combined", "coordinates.csv")
    
    search_paths = [coord_in_result, coord_in_parent, coord_in_combined]
    
    print(f"[数据加载器] 正在从以下路径加载预测结果: {predict_path}")
    print(f"[数据加载器] 正在搜索坐标文件，路径列表: {search_paths}")
    
    for coord_path in search_paths:
        if os.path.exists(coord_path):
            try:
                temp_coords = pd.read_csv(coord_path, index_col=0)
                print(f"[数据加载器] 找到坐标文件 {coord_path}，行数: {len(temp_coords)}，预测结果行数: {len(predict_df)}")
                
                # 宽松检查：验证索引是否显著重叠，而非严格长度匹配
                # 严格匹配常因细微的过滤差异而失败
                common_indices = predict_df.index.intersection(temp_coords.index)
                match_ratio = len(common_indices) / len(predict_df)
                
                if match_ratio > 0.9: # 允许 90% 的重叠度
                     coords = temp_coords.loc[predict_df.index] # 重新对齐
                     print(f"[数据加载器] 找到匹配的坐标！已对齐 {len(coords)} 行数据。")
                     break
                else:
                    print(f"[数据加载器] 索引不匹配。重叠率: {match_ratio:.2f}")

            except Exception as e:
                print(f"[数据加载器] 读取 {coord_path} 时出错: {e}")
                continue
    
    if coords is None:
        print("[数据加载器] 未能找到匹配的坐标文件。")
        
    return predict_df, coords

def get_cell_types(predict_df: pd.DataFrame) -> List[str]:
    """获取预测矩阵中的各细胞亚群名称集合。"""
    return predict_df.columns.tolist()


def load_from_uploaded_files(uploaded_files: list) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    云端数据管道加载逻辑。
    直接处理浏览器上传的字节流对象，并执行内存级索性对齐。
    """
    predict_df = None
    coords = None
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        
        try:
            uploaded_file.seek(0)  # 确保文件指针在开头
            if "predict" in filename and filename.endswith(".csv"):
                predict_df = pd.read_csv(uploaded_file, index_col=0)
                print(f"[数据加载器] 已从上传文件加载预测结果: {uploaded_file.name}")
            elif "coord" in filename and filename.endswith(".csv"):
                coords = pd.read_csv(uploaded_file, index_col=0)
                print(f"[数据加载器] 已从上传文件加载坐标: {uploaded_file.name}")
        except Exception as e:
            print(f"[数据加载器] 读取上传文件 {uploaded_file.name} 时出错: {e}")
            continue
    
    # 对齐索引
    if predict_df is not None and coords is not None:
        common_indices = predict_df.index.intersection(coords.index)
        match_ratio = len(common_indices) / len(predict_df)
        
        if match_ratio > 0.9:
            coords = coords.loc[predict_df.index]
            print(f"[数据加载器] 已从上传文件对齐 {len(coords)} 行数据。")
        else:
            print(f"[数据加载器] 上传文件索引不匹配。重叠率: {match_ratio:.2f}")
            coords = None  # 索引不匹配，丢弃坐标
    
    return predict_df, coords
