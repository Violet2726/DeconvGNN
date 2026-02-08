
import os
import io
import logging
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional, List, Dict, Any

# 数据目录配置
DATA_DIRS: Dict[str, str] = {}

def _get_env_int(key: str, default: int) -> int:
    """读取环境变量并转换为正整数，失败则回退默认值。"""
    try:
        value = int(os.getenv(key, default))
        return value if value > 0 else default
    except Exception:
        return default

def _get_env_float(key: str, default: float) -> float:
    """读取环境变量并转换为正浮点数，失败则回退默认值。"""
    try:
        value = float(os.getenv(key, default))
        return value if value > 0 else default
    except Exception:
        return default

def _get_logger() -> logging.Logger:
    """获取数据加载模块的日志记录器。"""
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
PARALLEL_LOAD_DEFAULT = _get_env_int("DECONV_VIS_PARALLEL_LOAD", 1) == 1
BINARY_CACHE_DEFAULT = _get_env_int("DECONV_VIS_USE_BINARY_CACHE", 1) == 1
SUMMARY_CACHE_DEFAULT = _get_env_int("DECONV_VIS_USE_SUMMARY_CACHE", 1) == 1
SUMMARY_SAMPLE_ROWS = _get_env_int("DECONV_VIS_SUMMARY_ROWS", 200)

def _normalize_coords(coords: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """统一坐标列为 x/y，兼容不同命名格式。"""
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
    """校验预测结果与坐标数据，并返回校验后的结果与提示信息。"""
    errors: List[str] = []
    warnings: List[str] = []
    # 预测结果基础校验
    if predict_df is None or predict_df.empty:
        errors.append("预测结果为空或无法读取")
        return None, None, errors, warnings
    if predict_df.index.has_duplicates:
        errors.append("预测结果索引存在重复值")
    # 检查非数值列
    non_numeric_cols = [
        col for col in predict_df.columns
        if not pd.api.types.is_numeric_dtype(predict_df[col])
    ]
    if non_numeric_cols:
        errors.append("预测结果包含非数值列")
    # 规范化坐标列
    coords = _normalize_coords(coords)
    # 坐标可用性校验
    if coords is None:
        warnings.append("坐标文件缺失或无法解析")
        return predict_df, None, errors, warnings
    if "x" not in coords.columns or "y" not in coords.columns:
        warnings.append("坐标列缺失或格式不兼容")
        return predict_df, None, errors, warnings
    if coords[["x", "y"]].isnull().any().any():
        warnings.append("坐标数据存在空值")
        return predict_df, None, errors, warnings
    # 索引重叠度检查
    common_indices = predict_df.index.intersection(coords.index)
    match_ratio = len(common_indices) / len(predict_df)
    if match_ratio < MATCH_RATIO_THRESHOLD:
        warnings.append(f"坐标与预测结果索引重叠不足 ({match_ratio:.2%})")
        return predict_df, None, errors, warnings
    # 对齐索引
    coords = coords.loc[predict_df.index]
    return predict_df, coords, errors, warnings

def _read_csv_path(path: str) -> pd.DataFrame:
    """从磁盘路径读取 CSV。"""
    return pd.read_csv(path, index_col=0, low_memory=False, engine="c", memory_map=True)

def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    """从字节流读取 CSV。"""
    return pd.read_csv(io.BytesIO(data), index_col=0, low_memory=False, engine="c")

def _cache_paths(result_dir: str) -> Tuple[str, str]:
    """生成预测结果与坐标的缓存路径。"""
    return (
        os.path.join(result_dir, "predict_result.pkl"),
        os.path.join(result_dir, "coordinates.pkl")
    )

def _summary_cache_path(result_dir: str) -> str:
    """生成摘要索引的缓存路径。"""
    return os.path.join(result_dir, "predict_summary.pkl")

def _is_cache_valid(cache_path: str, source_path: Optional[str]) -> bool:
    """判断缓存是否存在且未过期。"""
    if not source_path or not os.path.exists(cache_path) or not os.path.exists(source_path):
        return False
    try:
        return os.path.getmtime(cache_path) >= os.path.getmtime(source_path)
    except Exception:
        return False

def _read_pickle_path(path: str) -> pd.DataFrame:
    """读取 Pickle 缓存文件。"""
    return pd.read_pickle(path)

def _write_pickle_safe(path: str, df: Optional[pd.DataFrame]) -> None:
    """安全写入 Pickle 缓存。"""
    if df is None:
        return
    try:
        df.to_pickle(path)
    except Exception as exc:
        logger.warning("二进制缓存写入失败: %s", path, exc_info=exc)

def _count_csv_rows(path: str) -> int:
    """快速统计 CSV 行数（不含表头）。"""
    try:
        with open(path, "rb") as handle:
            count = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                count += chunk.count(b"\n")
        return max(count - 1, 0)
    except Exception as exc:
        logger.warning("行数统计失败: %s", path, exc_info=exc)
        return 0

def load_summary(
    result_dir: str,
    use_cache: bool = SUMMARY_CACHE_DEFAULT
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """生成或读取预测结果摘要索引。"""
    errors: List[str] = []
    # 检查源文件
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, ["未找到 predict_result.csv"]

    summary_cache = _summary_cache_path(result_dir)
    # 命中缓存则直接返回
    if use_cache and _is_cache_valid(summary_cache, predict_path):
        try:
            cached = pd.read_pickle(summary_cache)
            return cached, errors
        except Exception as exc:
            logger.warning("摘要缓存读取失败", exc_info=exc)

    try:
        # 采样读取并构建摘要
        row_count = _count_csv_rows(predict_path)
        sample_df = pd.read_csv(
            predict_path,
            index_col=0,
            nrows=SUMMARY_SAMPLE_ROWS,
            low_memory=False,
            engine="c"
        )
        summary = {
            "row_count": row_count,
            "column_count": len(sample_df.columns),
            "columns": sample_df.columns.tolist(),
            "sample": sample_df
        }
        # 写入缓存
        if use_cache:
            pd.to_pickle(summary, summary_cache)
        return summary, errors
    except Exception as exc:
        logger.warning("摘要索引生成失败", exc_info=exc)
        errors.append("摘要索引生成失败")
        return None, errors

def _load_results_core(
    result_dir: str,
    use_parallel: bool = PARALLEL_LOAD_DEFAULT,
    use_binary_cache: bool = BINARY_CACHE_DEFAULT
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    """
    数据集加载核心入口。
    从磁盘读取反卷积结果，并按层级策略查找关联坐标文件。
    """
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None, ["未找到 predict_result.csv"], []
    
    # 分级路径探测：结果目录 -> 数据集父目录 -> combined 目录
    coord_in_result = os.path.join(result_dir, "coordinates.csv")
    parent_dir = os.path.dirname(result_dir)
    coord_in_parent = os.path.join(parent_dir, "coordinates.csv")
    coord_in_combined = os.path.join(parent_dir, "combined", "coordinates.csv")
    
    search_paths = [coord_in_result, coord_in_parent, coord_in_combined]
    
    logger.info("加载预测结果路径: %s", predict_path)
    logger.info("坐标文件候选路径: %s", search_paths)

    coord_path = next((path for path in search_paths if os.path.exists(path)), None)
    predict_df = None
    coords = None
    predict_cache, coords_cache = _cache_paths(result_dir)

    # 命中预测结果缓存
    if use_binary_cache and _is_cache_valid(predict_cache, predict_path):
        try:
            predict_df = _read_pickle_path(predict_cache)
        except Exception as exc:
            logger.warning("预测结果缓存读取失败", exc_info=exc)
            predict_df = None

    # 命中坐标缓存
    if use_binary_cache and coord_path and _is_cache_valid(coords_cache, coord_path):
        try:
            coords = _read_pickle_path(coords_cache)
            if predict_df is not None:
                logger.info(
                    "坐标缓存读取成功: %s 行数=%s 预测行数=%s",
                    coords_cache,
                    len(coords),
                    len(predict_df)
                )
        except Exception as exc:
            logger.warning("坐标缓存读取失败", exc_info=exc)
            coords = None

    if predict_df is not None and (coords is not None or coord_path is None):
        # 缓存数据直接校验
        predict_df, coords, errors, warnings = validate_dataset(predict_df, coords)
        return predict_df, coords, errors, warnings

    if predict_df is None:
        # 预测结果与坐标并行读取
        if use_parallel and coord_path:
            with ThreadPoolExecutor(max_workers=2) as executor:
                predict_future = executor.submit(_read_csv_path, predict_path)
                coords_future = executor.submit(_read_csv_path, coord_path)
                try:
                    predict_df = predict_future.result()
                except Exception as exc:
                    logger.error("预测结果读取失败", exc_info=exc)
                    return None, None, ["预测结果读取失败"], []
                try:
                    coords = coords_future.result()
                    logger.info(
                        "坐标文件读取成功: %s 行数=%s 预测行数=%s",
                        coord_path,
                        len(coords),
                        len(predict_df)
                    )
                except Exception as exc:
                    logger.warning("坐标文件读取失败: %s", coord_path, exc_info=exc)
                    coords = None
        else:
            # 预测结果与坐标串行读取
            try:
                predict_df = _read_csv_path(predict_path)
            except Exception as exc:
                logger.error("预测结果读取失败", exc_info=exc)
                return None, None, ["预测结果读取失败"], []

            if coord_path:
                try:
                    coords = _read_csv_path(coord_path)
                    logger.info(
                        "坐标文件读取成功: %s 行数=%s 预测行数=%s",
                        coord_path,
                        len(coords),
                        len(predict_df)
                    )
                except Exception as exc:
                    logger.warning("坐标文件读取失败: %s", coord_path, exc_info=exc)
                    coords = None
    elif coords is None and coord_path:
        # 已有预测结果，仅补充坐标
        try:
            coords = _read_csv_path(coord_path)
            logger.info(
                "坐标文件读取成功: %s 行数=%s 预测行数=%s",
                coord_path,
                len(coords),
                len(predict_df)
            )
        except Exception as exc:
            logger.warning("坐标文件读取失败: %s", coord_path, exc_info=exc)
            coords = None
    
    if coords is None:
        logger.warning("未找到可用的坐标文件")

    if use_binary_cache:
        # 更新二进制缓存
        _write_pickle_safe(predict_cache, predict_df)
        _write_pickle_safe(coords_cache, coords)

    predict_df, coords, errors, warnings = validate_dataset(predict_df, coords)
    return predict_df, coords, errors, warnings

@st.cache_data(ttl=CACHE_TTL, max_entries=CACHE_MAX_ENTRIES, show_spinner=False)
def load_results(
    result_dir: str,
    use_parallel: bool = PARALLEL_LOAD_DEFAULT,
    use_binary_cache: bool = BINARY_CACHE_DEFAULT
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    """加载数据集并返回预测结果与坐标。"""
    return _load_results_core(result_dir, use_parallel, use_binary_cache)

def prewarm_cache(
    result_dir: str,
    use_parallel: bool,
    use_binary_cache: bool
) -> Tuple[List[str], List[str]]:
    """预热缓存并返回潜在的错误与警告信息。"""
    _, _, errors, warnings = _load_results_core(result_dir, use_parallel, use_binary_cache)
    return errors, warnings

def get_cell_types(predict_df: pd.DataFrame) -> List[str]:
    """获取预测矩阵中的细胞类型名称列表。"""
    return predict_df.columns.tolist()


def load_from_uploaded_files(
    uploaded_files: list,
    use_parallel: bool = PARALLEL_LOAD_DEFAULT
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[str], List[str]]:
    """
    云端上传数据加载逻辑。
    解析浏览器上传字节流，并完成内存级对齐与校验。
    """
    predict_df = None
    coords = None
    
    predict_file = None
    coords_file = None

    # 识别上传文件类型
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        if "predict" in filename and filename.endswith(".csv"):
            predict_file = uploaded_file
        elif "coord" in filename and filename.endswith(".csv"):
            coords_file = uploaded_file

    if use_parallel and predict_file and coords_file:
        # 并行读取预测结果与坐标
        try:
            predict_bytes = predict_file.getvalue()
            coords_bytes = coords_file.getvalue()
        except Exception:
            predict_file.seek(0)
            coords_file.seek(0)
            predict_bytes = predict_file.read()
            coords_bytes = coords_file.read()

        with ThreadPoolExecutor(max_workers=2) as executor:
            predict_future = executor.submit(_read_csv_bytes, predict_bytes)
            coords_future = executor.submit(_read_csv_bytes, coords_bytes)
            try:
                predict_df = predict_future.result()
                logger.info("上传文件加载预测结果: %s", predict_file.name)
            except Exception as exc:
                logger.warning("上传文件读取失败: %s", predict_file.name, exc_info=exc)
                predict_df = None
            try:
                coords = coords_future.result()
                logger.info("上传文件加载坐标: %s", coords_file.name)
            except Exception as exc:
                logger.warning("上传文件读取失败: %s", coords_file.name, exc_info=exc)
                coords = None
    else:
        # 串行读取上传文件
        for uploaded_file in uploaded_files:
            filename = uploaded_file.name.lower()
            try:
                uploaded_file.seek(0)
                if "predict" in filename and filename.endswith(".csv"):
                    predict_df = _read_csv_bytes(uploaded_file.read())
                    logger.info("上传文件加载预测结果: %s", uploaded_file.name)
                elif "coord" in filename and filename.endswith(".csv"):
                    coords = _read_csv_bytes(uploaded_file.read())
                    logger.info("上传文件加载坐标: %s", uploaded_file.name)
            except Exception as exc:
                logger.warning("上传文件读取失败: %s", uploaded_file.name, exc_info=exc)
                continue

    # 统一校验与对齐
    predict_df, coords, errors, warnings = validate_dataset(predict_df, coords)
    return predict_df, coords, errors, warnings
