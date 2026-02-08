
import os
import logging
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List, Dict

# 数据目录配置 (初始为空)
DATA_DIRS: Dict[str, str] = {}

def _get_env_int(key: str, default: int) -> int:
    try:
        value = int(os.getenv(key, default))
        return value if value > 0 else default
    except Exception:
        return default

def _get_env_float(key: str, default: float) -> float:
    try:
        value = float(os.getenv(key, default))
        return value if value > 0 else default
    except Exception:
        return default

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("visualization.data_loader")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("DECONV_VIS_LOG_LEVEL", "INFO").upper())
    return logger

logger = _get_logger()
CACHE_TTL = _get_env_int("DECONV_VIS_CACHE_TTL", 3600)
CACHE_MAX_ENTRIES = _get_env_int("DECONV_VIS_CACHE_MAX_ENTRIES", 10)
MATCH_RATIO_THRESHOLD = _get_env_float("DECONV_VIS_MATCH_RATIO_THRESHOLD", 0.9)

def _normalize_coords(coords: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if coords is None:
        return None
    if "x" in coords.columns and "y" in coords.columns:
        return coords
    if "coor_X" in coords.columns and "coor_Y" in coords.columns:
        return coords.rename(columns={"coor_X": "x", "coor_Y": "y"})
    if coords.shape[1] >= 2:
        normalized = coords.iloc[:, :2].copy()
        normalized.columns = ["x", "y"]
        return normalized
    return coords

def validate_dataset(
    predict_df: Optional[pd.DataFrame],
    coords: Optional[pd.DataFrame]
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if predict_df is None or predict_df.empty:
        errors.append("预测结果为空或无法读取")
        return None, None, errors, warnings
    if predict_df.index.has_duplicates:
        errors.append("预测结果索引存在重复值")
    non_numeric_cols = [
        col for col in predict_df.columns
        if not pd.api.types.is_numeric_dtype(predict_df[col])
    ]
    if non_numeric_cols:
        errors.append("预测结果包含非数值列")
    coords = _normalize_coords(coords)
    if coords is None:
        warnings.append("坐标文件缺失或无法解析")
        return predict_df, None, errors, warnings
    if "x" not in coords.columns or "y" not in coords.columns:
        warnings.append("坐标列缺失或格式不兼容")
        return predict_df, None, errors, warnings
    if coords[["x", "y"]].isnull().any().any():
        warnings.append("坐标数据存在空值")
        return predict_df, None, errors, warnings
    common_indices = predict_df.index.intersection(coords.index)
    match_ratio = len(common_indices) / len(predict_df)
    if match_ratio < MATCH_RATIO_THRESHOLD:
        warnings.append(f"坐标与预测结果索引重叠不足 ({match_ratio:.2%})")
        return predict_df, None, errors, warnings
    coords = coords.loc[predict_df.index]
    return predict_df, coords, errors, warnings

@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def load_results(result_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    """
    数据集加载核心引擎。
    从物理磁盘读取反卷积结果，并执行分级探测策略搜索关联的空间位点坐标文件。
    """
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None, ["未找到 predict_result.csv"], []
    
    try:
        predict_df = pd.read_csv(predict_path, index_col=0)
    except Exception as exc:
        logger.error("预测结果读取失败", exc_info=exc)
        return None, None, ["预测结果读取失败"], []
    
    # 尝试加载坐标
    coords = None
    
    # 执行分级路径探测机制：结果目录 -> 数据集父目录 -> combined 合并目录
    coord_in_result = os.path.join(result_dir, "coordinates.csv")
    parent_dir = os.path.dirname(result_dir)
    coord_in_parent = os.path.join(parent_dir, "coordinates.csv")
    coord_in_combined = os.path.join(parent_dir, "combined", "coordinates.csv")
    
    search_paths = [coord_in_result, coord_in_parent, coord_in_combined]
    
    logger.info("加载预测结果路径: %s", predict_path)
    logger.info("坐标文件候选路径: %s", search_paths)
    
    for coord_path in search_paths:
        if os.path.exists(coord_path):
            try:
                temp_coords = pd.read_csv(coord_path, index_col=0)
                logger.info(
                    "坐标文件读取成功: %s 行数=%s 预测行数=%s",
                    coord_path,
                    len(temp_coords),
                    len(predict_df)
                )
                coords = temp_coords
                break

            except Exception as exc:
                logger.warning("坐标文件读取失败: %s", coord_path, exc_info=exc)
                continue
    
    if coords is None:
        logger.warning("未找到可用的坐标文件")

    predict_df, coords, errors, warnings = validate_dataset(predict_df, coords)
    return predict_df, coords, errors, warnings

def get_cell_types(predict_df: pd.DataFrame) -> List[str]:
    """获取预测矩阵中的各细胞亚群名称集合。"""
    return predict_df.columns.tolist()


def load_from_uploaded_files(uploaded_files: list) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    """
    云端数据管道加载逻辑。
    直接处理浏览器上传的字节流对象，并执行内存级索性对齐。
    """
    predict_df = None
    coords = None
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        
        try:
            uploaded_file.seek(0)
            if "predict" in filename and filename.endswith(".csv"):
                predict_df = pd.read_csv(uploaded_file, index_col=0)
                logger.info("上传文件加载预测结果: %s", uploaded_file.name)
            elif "coord" in filename and filename.endswith(".csv"):
                coords = pd.read_csv(uploaded_file, index_col=0)
                logger.info("上传文件加载坐标: %s", uploaded_file.name)
        except Exception as exc:
            logger.warning("上传文件读取失败: %s", uploaded_file.name, exc_info=exc)
            continue

    predict_df, coords, errors, warnings = validate_dataset(predict_df, coords)
    return predict_df, coords, errors, warnings
