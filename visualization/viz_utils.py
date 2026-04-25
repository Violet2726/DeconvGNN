# -*- coding: utf-8 -*-
"""
DeconvGNN-Vis 图表生成与可视化辅助函数。

该模块把反卷积结果转换为 Plotly/Matplotlib 图表。核心设计是：用
Matplotlib 批量渲染高分辨率饼图背景，再用 Plotly WebGL 提供交互层，
从而在上万空间位点上兼顾细节展示和浏览器端性能。
"""
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import os
import logging
from PIL import Image
import pandas as pd
import hashlib
from functools import lru_cache

from typing import Tuple, List, Optional, Any, Dict, Callable
from matplotlib.axes import Axes
import plotly.graph_objects as go
from pathlib import Path

# 运行时环境检测与 Streamlit 兼容层。该文件也会被离线脚本导入，
# 因此不能强依赖 Streamlit 存在。
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

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
    """获取可视化工具模块的日志记录器。"""
    logger = logging.getLogger("visualization.viz_utils")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("DECONV_VIS_LOG_LEVEL", "INFO").upper())
    return logger

logger = _get_logger()
TOP_N_CATEGORIES = _get_env_int("DECONV_VIS_TOP_N_CATEGORIES", 4)
LOD_THRESHOLD = _get_env_int("DECONV_VIS_LOD_THRESHOLD", 5000)
LOD_SAMPLE_RATIO = _get_env_float("DECONV_VIS_LOD_SAMPLE_RATIO", 0.3)
CHART_CACHE_SIZE = _get_env_int("DECONV_VIS_CHART_CACHE_SIZE", 16)

def is_cloud_environment() -> bool:
    """
    判断是否运行在 Streamlit Cloud 环境，用于区分本地与云端流程。

    Streamlit Cloud 不适合打开本地文件夹选择器，因此上传/导入逻辑会根据
    该判断走不同分支。
    """
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") is not None or
        os.environ.get("IS_STREAMLIT_CLOUD") is not None or
        os.path.exists("/mount/src")  # Streamlit Cloud 容器特有路径
    )

def get_data_fingerprint(df: pd.DataFrame) -> str:
    """
    生成 DataFrame 的轻量指纹，用于缓存键。
    采用形状与前后样本行，避免全量哈希成本。

    注意:
        该指纹用于界面缓存失效判断，不用于安全校验或精确文件完整性验证。
    """
    shape_str = f"{df.shape}"
    # 取前5行和后5行的字符串表示
    sample_str = df.head(5).to_string() + df.tail(5).to_string()
    return hashlib.md5((shape_str + sample_str).encode()).hexdigest()[:12]

# 缓存装饰器：Streamlit 使用 st.cache_data，否则回退 lru_cache。
def cached_chart(func):
    """根据运行环境选择 Streamlit 或 LRU 缓存装饰器。"""
    if HAS_STREAMLIT:
        return st.cache_data(ttl=1800, max_entries=CHART_CACHE_SIZE, show_spinner=False)(func)
    else:
        return lru_cache(maxsize=CHART_CACHE_SIZE)(func)

# 资源路径配置
ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.png"

def get_base64_image(image_path: str) -> str:
    """读取图片并返回 Base64 编码字符串。"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

@cached_chart
def get_base64_image_cached(image_path: str) -> str:
    """缓存图片 Base64 编码结果，减少重复 IO。"""
    return get_base64_image(image_path)


def get_adaptive_dpi(n_points: int) -> int:
    """
    按数据规模动态调整 DPI，平衡清晰度与计算开销。

    数据点越多，单个饼图在屏幕上越小，继续提高 DPI 的视觉收益有限；
    因此大数据集降低 DPI 可以显著减少渲染时间和图片体积。
    """
    if n_points > 10000: return 150
    elif n_points > 5000: return 300
    elif n_points > 2000: return 450
    else: return 600

def apply_lod_sampling(predict_df: pd.DataFrame, coords: pd.DataFrame, 
                       force_full: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    对超大规模数据集应用 LOD 采样。

    LOD（Level of Detail）只影响交互散点层，不改变用于生成背景饼图的
    全量数据。这样缩放/悬停更流畅，同时保留空间组成的整体视觉信息。
    """
    n_points = len(predict_df)
    
    if force_full or n_points <= LOD_THRESHOLD:
        return predict_df, coords, False
    
    # 执行随机均匀下采样
    sample_n = int(n_points * LOD_SAMPLE_RATIO)
    sample_indices = np.random.choice(predict_df.index, size=sample_n, replace=False)
    
    return predict_df.loc[sample_indices], coords.loc[sample_indices], True

def generate_clean_pie_chart(predict_df, coords, point_size=20, 
                              progress_callback: Optional[Callable[[float, str], None]] = None):
    """
    高性能饼图背景生成器。
    使用 Matplotlib PatchCollection 批量化渲染数万个微型饼图，生成透明背景的高清 PNG。
    该图片随后会被嵌入 Plotly 面板作为图层背景。

    参数:
        predict_df: 细胞类型比例矩阵，行为空间位点，列为细胞类型。
        coords: 与预测矩阵索引对齐的坐标表，需包含 `x` 和 `y`。
        point_size: 兼容参数；当前半径主要根据空间点间距自适应计算。
        progress_callback: 可选进度回调，供 Streamlit 进度条更新。

    返回:
        tuple: `(PIL.Image, (xlim, ylim))`，分别是透明背景图和绘图边界。
    """
    def update_progress(pct, msg):
        """转发进度到外部回调函数。"""
        if progress_callback:
            progress_callback(pct, msg)
    
    update_progress(0.05, "准备颜色映射...")
    
    # 准备颜色映射。颜色顺序可能基于细胞类型空间相关性聚类，
    # 让共定位模式相近的细胞类型在图例上更靠近。
    labels = predict_df.columns.tolist()
    color_map = get_color_map_cached(tuple(labels), predict_df)
    colors = [color_map[label] for label in labels]
    
    update_progress(0.1, "计算画布尺寸...")
    
    # 计算画布尺寸。保持真实坐标的宽高比，避免组织结构被拉伸。
    x_range = coords['x'].max() - coords['x'].min()
    y_range = coords['y'].max() - coords['y'].min()
    if y_range == 0: y_range = 1
    aspect_ratio = x_range / y_range
    
    # 自适应 DPI
    n_points = len(predict_df)
    dpi = get_adaptive_dpi(n_points)
    base_size = 12
    fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size), dpi=dpi)
    
    update_progress(0.15, f"使用 DPI={dpi} (数据点: {n_points})...")
    
    update_progress(0.2, "计算最佳点大小...")
    
    # 位点间距估计：取每个点最近邻距离的中位数作为典型 spot 间距，
    # 再按固定比例设置饼图半径，避免在高密度区域过度重叠。
    from sklearn.neighbors import NearestNeighbors
    coords_array = np.column_stack((coords['x'], coords['y']))
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords_array)
    distances, _ = nbrs.kneighbors(coords_array)
    avg_spacing = np.median(distances[:, 1]) # 获取平均步长
    
    update_progress(0.25, "正在构建图形集合 (PatchCollection)...")
    
    # 使用 PatchCollection 提升性能：一次性把所有 Wedge 加入 Axes，
    # 比逐个调用 ax.add_patch 快很多。
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection

    # 半径系数
    radius = avg_spacing * 0.42
    
    logger.info("使用 PatchCollection 批量渲染 DPI=%s 半径=%.4f", dpi, radius)
    
    wedges = []
    
    # 预计算数据
    predict_values = predict_df.values
    x_coords = coords['x'].values
    y_coords = coords['y'].values
    
    # 全局 Top N：每个点只绘制占比最高的若干细胞类型。
    # 这样可以减少扇区数量，也能突出主要组成。
    top_n = TOP_N_CATEGORIES
    total_points = len(predict_df)
    
    update_progress(0.3, f"构建 {total_points} 个饼图对象...")

    for i in range(total_points):
        # 更新进度（每 10% 一次）
        if i % max(1, total_points // 10) == 0:
            progress = 0.3 + 0.5 * (i / total_points)
            update_progress(progress, f"处理点 {i}/{total_points}...")
        # 获取分布与坐标
        dist = predict_values[i]
        xc, yc = x_coords[i], y_coords[i]
        
        # 计算该位点的 Top N 细胞类型索引。
        if np.all(dist == 0): continue
        
        sorted_indices = np.argsort(dist)[::-1]
        top_indices = sorted_indices[:top_n]
        
        current_vals = []
        current_colors = []
        for idx in top_indices:
            if dist[idx] > 0:
                current_vals.append(dist[idx])
                current_colors.append(colors[idx])
        
        if not current_vals: continue
        
        # 对 Top N 重新归一化后构建扇形。这里展示的是 Top N 内部构成，
        # 不是全量细胞类型的绝对总和。
        total = sum(current_vals)
        # 起始角度（0 度对应 X 轴正向）
        current_angle = 0.0
        
        for val, color in zip(current_vals, current_colors):
            # 计算扇区跨度
            ratio = val / total
            theta = ratio * 360.0
            
            # 创建扇形 Wedge((x,y), r, start, end)
            # Wedge 使用度数
            w = Wedge((xc, yc), radius, current_angle, current_angle + theta, facecolor=color, linewidth=0)
            wedges.append(w)
            
            current_angle += theta

    # 批量添加到 Axes
    if wedges:
        logger.info("渲染扇形图层 count=%s", len(wedges))
        # match_original=True 保留每个 Wedge 的颜色
        p = PatchCollection(wedges, match_original=True)
        ax.add_collection(p)
        
    # 坐标范围与 Plotly 对齐，并增加少量边距避免边缘饼图被裁切。
    padding = x_range * 0.05
    ax.set_xlim(coords['x'].min() - padding, coords['x'].max() + padding)
    ax.set_ylim(coords['y'].min() - padding, coords['y'].max() + padding)
    
    # 隐藏坐标装饰
    ax.axis('off')
    plt.margins(0)
    # 去除边距
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    # 透明背景
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # 保存到内存
    update_progress(0.9, "保存图像...")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    update_progress(1.0, "完成！")
    return Image.open(buf), (ax.get_xlim(), ax.get_ylim())

def get_color_map(labels: List[str], predict_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    全局颜色映射表生成器。
    
    [Co-localization 配色引擎]:
    如果提供了 predict_df，则通过 Ward 凝聚层次聚类分析细胞类型间的空间相关性。
    在色相环 (HSL Cycle) 上为空间分布相近的细胞分配相邻颜色，从而揭示微环境特征。

    参数:
        labels: 细胞类型名称列表。
        predict_df: 可选预测矩阵，用于基于空间相关性调整颜色顺序。

    返回:
        dict: `{细胞类型: 十六进制颜色}`。
    """
    import matplotlib.colors as mcolors
    import colorsys
    import numpy as np
    import pandas as pd
    
    # 默认按字母序，保证确定性
    sorted_labels = sorted(list(set(labels)))
    
    # 基于空间相关性的层次聚类排序：相似的细胞类型在图例上相邻，
    # 帮助用户更快识别共定位或互斥模式。
    if predict_df is not None:
        try:
            import scipy.cluster.hierarchy as sch
            import scipy.spatial.distance as dist
            
            # 1. 确保 predict_df 仅包含需要的列
            valid_cols = [c for c in sorted_labels if c in predict_df.columns]
            if len(valid_cols) > 2:
                data = predict_df[valid_cols]
                
                # 2. 计算细胞类型之间的相关性矩阵。聚类对象是“细胞类型”，
                #    每个类型的特征向量是其在所有空间位点上的比例分布。
                correlation_matrix = data.corr().fillna(0)
                
                # 3. 使用 Ward 方法进行层次聚类，获得稳定且可解释的叶节点顺序。
                d = sch.distance.pdist(correlation_matrix)
                linkage = sch.linkage(d, method='ward')
                
                # 4. 获取最优叶节点排序索引
                ind = sch.leaves_list(linkage)
                
                # 根据聚类叶序重排标签
                sorted_labels = [correlation_matrix.columns[i] for i in ind]
        except Exception as exc:
            logger.warning("颜色聚类排序失败，回退到字母序", exc_info=exc)
            
    n = len(sorted_labels)
    colors = []
    
    # 生成均匀分布的色相 (Hue)
    # 使用 ggplot2/Seurat 风格的 HSL 色轮
    hues = np.linspace(0, 1, n + 1)[:-1] 
    
    for h in hues:
        # H=Hue
        # L=0.5 (亮度)
        # S=0.65 (饱和度)
        rgb = colorsys.hls_to_rgb(h, 0.5, 0.65)
        hex_color = mcolors.to_hex(rgb)
        colors.append(hex_color)
    
    return dict(zip(sorted_labels, colors))

@cached_chart
def get_color_map_cached(labels: Tuple[str, ...], predict_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """缓存颜色映射以减少重复计算。"""
    return get_color_map(list(labels), predict_df)


def generate_plotly_scatter(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame, 
                          hover_count: int, bg_img: Any, bounds: Tuple[float, float], 
                          color_map: Dict[str, str]) -> go.Figure:
    """
    核心图表插件：空间组分交互散点图 (Tab 1)。
    采用多层渲染技术：底层为 Matplotlib 计算的饼图纹理，顶层为高性能投影交互层。

    返回:
        go.Figure: 可直接传入 `st.plotly_chart` 的 Plotly 图表对象。
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    xlim, ylim = bounds
    plot_df = coords_for_plot.copy()
    
    # 优化悬停文本生成：使用 numpy 批量排序，避免 pandas 逐行 apply。
    vals = predict_df.values
    cols = predict_df.columns.tolist()
    
    # 获取每一行前 N 个最大元素的索引
    # np.argsort 是升序排序，所以取最后 hover_count 个元素并反转
    top_n_indices = np.argsort(vals, axis=1)[:, -hover_count:][:, ::-1]
    
    hover_texts = []
    indices = predict_df.index.tolist()
    
    for i in range(len(predict_df)):
        idx_label = indices[i]
        text = f"<b>位置 {idx_label}</b><br>"
        
        # 确定该行的 Top N
        row_indices = top_n_indices[i]
        
        for col_idx in row_indices:
            proportion = vals[i, col_idx]
            if proportion > 0: # 仅显示非零比例
                cell_type = cols[col_idx]
                text += f"{cell_type}: {proportion:.2%}<br>"
                
        hover_texts.append(text)
        
    plot_df['hover_text'] = hover_texts
    
    fig = px.scatter(
        plot_df, x='x', y='y',
        hover_name='hover_text',
        title='空间组成分布',
        render_mode='webgl'  # 启用 WebGL 渲染，大幅降低内存
    )
    
    fig.update_traces(
        marker=dict(opacity=0),
        hovertemplate='%{hovertext}<extra></extra>'
    )
    
    # 虚拟图例：背景饼图不是 Plotly trace，因此需要添加空散点来生成可读图例。
    for cell_type, color in color_map.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color, symbol='circle'),
                name=cell_type,
                showlegend=True
            )
        )
    
    # 添加背景饼图图片，并使用相同坐标边界保证和透明交互层完全对齐。
    if bg_img:
        fig.add_layout_image(
            dict(
                source=bg_img,
                xref="x", yref="y",
                x=xlim[0], y=ylim[1],
                sizex=xlim[1] - xlim[0],
                sizey=ylim[1] - ylim[0],
                sizing="stretch",
                layer="below"
            )
        )
    
    fig.update_xaxes(range=[xlim[0], xlim[1]], visible=False, showgrid=False)
    fig.update_yaxes(range=[ylim[0], ylim[1]], visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
    
    fig.update_layout(
        height=800,
        autosize=True,  # 恢复自动调整
        margin=dict(l=0, r=200, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title="细胞类型 (饼图颜色)",
            orientation="v", 
            yanchor="top", y=1, 
            xanchor="left", x=1.02,
            itemclick=False, itemdoubleclick=False
        ),
        dragmode='pan'
    )
    return fig


def generate_dominant_scatter(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame,
                             hover_count: int, color_map: Dict[str, str]) -> go.Figure:
    """
    核心图表插件：优势细胞亚群映射图 (Tab 2)。
    基于 WebGL 渲染，支持位点大小与数据规模的自适应缩放。

    每个点只显示预测占比最高的细胞类型，适合快速观察空间区域的主导亚群。
    """
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['主要细胞类型'] = predict_df.idxmax(axis=1).values
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    # 自适应点大小。启发式公式假设画布有效区域约 600-800px，
    # 点数越多则点越小，避免大数据集出现严重遮挡。
    n_points = len(predict_df)
    adaptive_size = max(3, (550 / np.sqrt(n_points)))
    adaptive_size = min(15, adaptive_size) # 限制最大值
    
    display_df['pixel_size'] = adaptive_size
    
    logger.info("自适应点大小 %.2f 点数=%s", adaptive_size, n_points)
        
    # 修改：不再强制字母排序，而是跟随 color_map 的键顺序（即聚类顺序）
    # unique_types = sorted(predict_df.columns.tolist())
    unique_types = [t for t in color_map.keys() if t in predict_df.columns]
    
    fig = go.Figure()
    
    for cell_type in unique_types:
        subset = display_df[display_df['主要细胞类型'] == cell_type]
        if len(subset) == 0: continue
            
        # 优化悬停文本生成：按当前细胞类型子集批量计算 Top N，
        # 避免对全量预测矩阵重复排序。
        
        # 在子集上使用高效处理方法
        subset_indices = subset.index
        subset_predict_vals = predict_df.loc[subset_indices].values
        subset_cols = predict_df.columns.tolist()
        
        # 批量排序获取 Top N
        sub_top_res = np.argsort(subset_predict_vals, axis=1)[:, -hover_count:][:, ::-1]
        
        hover_texts = []
        for i, idx_val in enumerate(subset_indices):
            major_type = cell_type  # 当前循环已确定主导细胞类型。
            major_prop = subset.loc[idx_val, '主要比例']
            
            text = f"<b>位置 {idx_val}</b><br>主要类型: {major_type} ({major_prop:.2%})<br>"
            
            row_indices = sub_top_res[i]
            for col_idx in row_indices:
                proportion = subset_predict_vals[i, col_idx]
                if proportion > 0:
                     ct = subset_cols[col_idx]
                     text += f"{ct}: {proportion:.2%}<br>"
            hover_texts.append(text)

        fig.add_trace(go.Scattergl(
            x=subset['x'], y=subset['y'],
            mode='markers',
            name=cell_type,
            marker=dict(
                color=color_map.get(cell_type, '#333'),
                size=subset['pixel_size'],
                sizemode='diameter',
                opacity=0.9,
                line=dict(width=0)
            ),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hover_texts
        ))
        
    fig.update_layout(
        height=800, 
        autosize=True, # 恢复自动调整
        title='主要类型分布',
        margin=dict(l=0, r=150, t=30, b=0), # 显式减少顶部和左侧留白
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, showgrid=False),
        xaxis=dict(visible=False, showgrid=False),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode='pan',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def generate_proportion_bar(predict_df: pd.DataFrame) -> go.Figure:
    """
    为 Tab 3 生成各细胞类型平均占比柱状图。

    返回:
        go.Figure: 横向柱状图，按平均占比升序排列。
    """
    import plotly.express as px
    mean_proportions = predict_df.mean().sort_values(ascending=True)
    fig = px.bar(
        x=mean_proportions.values,
        y=mean_proportions.index,
        orientation='h',
        labels={'x': '平均比例', 'y': '细胞类型'},
        color=mean_proportions.values,
        color_continuous_scale='Blues',
        title="各细胞类型平均占比"
    )
    # 根据条目数量动态调整高度，防止细胞类型较多时标签拥挤。
    dynamic_height = max(500, len(mean_proportions) * 25)
    
    fig.update_layout(
        height=dynamic_height, 
        showlegend=False,
        # 增加左侧边距显示标签，增加右侧边距显示 Colorbar
        margin=dict(l=150, r=100),
        # 修复 Colorbar 字体颜色
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(color='white'),
                title=dict(font=dict(color='white'))
            )
        )
    )
    return fig


def generate_heatmap(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame, 
                    selected_type: str) -> go.Figure:
    """
    为 Tab 4 生成单个细胞类型的空间热图。

    参数:
        selected_type: 用户选中的细胞类型列名。
    """
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['proportion'] = predict_df[selected_type].values
    
    hover_texts = [f"<b>位置 {idx}</b><br>类型: {selected_type}<br>比例: {val:.2%}" 
                  for idx, val in zip(display_df.index, display_df['proportion'])]

    
    # 自适应点大小与优势亚群图保持一致，保证不同 Tab 的空间尺度感接近。
    n_points = len(predict_df)
    adaptive_size = max(3, (550 / np.sqrt(n_points)))
    adaptive_size = min(15, adaptive_size)
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=display_df['x'], y=display_df['y'],
        mode='markers',
        marker=dict(
            size=adaptive_size,
            color=display_df['proportion'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="比例",
                    font=dict(color='white')
                ),
                tickfont=dict(color='white'),
                ticks='',   # 去掉刻度线
                len=0.9
            ),
            opacity=1.0
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'单细胞类型热图: {selected_type}',
        height=800, autosize=True,
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, showgrid=False),
        xaxis=dict(visible=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        dragmode='pan', margin=dict(l=0, r=121.5, t=30, b=0),
    )
    return fig

def save_pie_chart_background(img: Image.Image, xlim: float, ylim: float, result_dir: str) -> None:
    """
    将实时渲染的背景图及其元数据持久化至磁盘，加速后续加载。

    元数据保存坐标边界，确保下次读取图片时仍能与 Plotly 坐标系对齐。
    """
    import os
    import json
    
    target_img = os.path.join(result_dir, "interactive_pie_background.png")
    target_meta = os.path.join(result_dir, "interactive_pie_bounds.json")
    
    try:
         img.save(target_img)
         with open(target_meta, 'w') as f:
             json.dump({'xlim': xlim, 'ylim': ylim}, f)
    except Exception as exc:
         logger.warning("无法保存背景缓存", exc_info=exc)

def open_folder_dialog() -> Optional[str]:
    """调取操作系统原生文件夹选择对话框（仅限本地环境）。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        # 设置根窗口，隐藏并置顶
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # 打开对话框
        folder = filedialog.askdirectory(title="选择 STdGCN 输出目录")
        
        # 清理
        root.destroy()
        return folder if folder else None
    except Exception as exc:
        logger.warning("打开文件夹对话框出错", exc_info=exc)
        return None

def generate_and_save_interactive_assets(predict_df, coordinates, output_dir):
    """
    生成并保存交互式可视化资源 (Top 4 饼图背景)。
    自动处理坐标映射并保存为 interactive_pie_background.png。

    该函数主要供训练结束后的离线脚本调用，提前生成背景图可以显著减少
    Streamlit 首次打开某个结果目录时的等待时间。
    """
    import os
    
    logger.info("正在生成交互式可视化背景图 Top=%s", TOP_N_CATEGORIES)
    
    # 确保坐标列名匹配。生成函数期望 coordinates 有 `x`、`y` 两列。
    coords = coordinates.copy()
    if 'coor_X' in coords.columns:
        coords = coords.rename(columns={'coor_X': 'x', 'coor_Y': 'y'})
    elif 'x' not in coords.columns:
        # 假设前两列是 x, y
        coords.columns = ['x', 'y']
        
    # 确保索引对齐，防止预测比例与坐标错位。
    common_index = predict_df.index.intersection(coords.index)
    if len(common_index) < len(predict_df):
        logger.warning("坐标与预测结果索引不完全匹配 交集=%s", len(common_index))
    
    predict_df = predict_df.loc[common_index]
    coords = coords.loc[common_index]

    # 生成图片。point_size=None 表示使用坐标间距自适应半径。
    try:
        img, (xlim, ylim) = generate_clean_pie_chart(predict_df, coords, point_size=None)
        
        # 保存到结果目录，供 Streamlit 后续直接读取。
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        save_pie_chart_background(img, xlim, ylim, output_dir)
        logger.info("交互式背景图已保存: %s", output_dir)
    except Exception as exc:
        logger.error("生成可视化资源时出错", exc_info=exc)

def handle_visualization_generation(paths):
    """
    可视化生成流程的单一入口函数：读取数据 -> 生成背景 -> 保存资源。

    训练脚本在模型输出后调用该函数，做到“训练结果可直接被 Web 页面加载”。
    """
    import os
    import pandas as pd
    
    logger.info("开始生成交互式可视化资源")
    
    res_path = os.path.join(paths['output_path'], 'predict_result.csv')
    coor_path = os.path.join(paths['ST_path'], 'coordinates.csv')
    
    if os.path.exists(res_path) and os.path.exists(coor_path):
        try:
            # 读取预测结果和坐标。
            pred_df = pd.read_csv(res_path, index_col=0)
            # 坐标文件在项目内统一为标准 CSV，这里按首列索引读取。
            coord_df = pd.read_csv(coor_path, index_col=0)
            
            # 调用生成函数并写入 output_path。
            generate_and_save_interactive_assets(pred_df, coord_df, paths['output_path'])
            logger.info("可视化资源生成完成")
        except Exception as exc:
            logger.error("可视化生成失败", exc_info=exc)
    else:
        logger.warning("找不到结果文件或坐标文件，跳过生成")
        logger.warning("预测结果: %s", res_path if os.path.exists(res_path) else f"{res_path} 缺失")
        logger.warning("坐标文件: %s", coor_path if os.path.exists(coor_path) else f"{coor_path} 缺失")

def get_or_generate_pie_background(predict_df: pd.DataFrame, coords: pd.DataFrame, 
                                 result_dir: str, 
                                 progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[Any, Tuple[float, float]]:
    """
    可视化资产核心调度器。
    
    工作流：
    1. 扫描目标目录是否存在有效的预计算资产 (.png & .json)。
    2. 若命中，执行毫秒级磁盘读取。
    3. 若未命中，则唤起并行渲染流水线生成新资产，并视环境执行自动保存行为。

    返回:
        tuple: `(bg_img, (xlim, ylim))`，用于嵌入 Plotly 背景图层。
    """
    import json
    from PIL import Image
    
    # 路径定义
    precomputed_img_path = os.path.join(result_dir, "interactive_pie_background.png")
    precomputed_meta_path = os.path.join(result_dir, "interactive_pie_bounds.json")
    
    # 1. 尝试从磁盘读取（仅对本地数据集有效）。
    if result_dir != "__UPLOADED__" and os.path.exists(precomputed_img_path) and os.path.exists(precomputed_meta_path):
        try:
            bg_img = Image.open(precomputed_img_path)
            with open(precomputed_meta_path, 'r') as f:
                metadata = json.load(f)
                return bg_img, (metadata['xlim'], metadata['ylim'])
        except Exception as exc:
            logger.warning("读取缓存背景失败，将重新生成", exc_info=exc)
            
    # 2. 未命中缓存时现场生成。
    bg_img, (xlim, ylim) = generate_clean_pie_chart(
        predict_df, coords, None, 
        progress_callback=progress_callback
    )
    
    # 3. 保存到磁盘（仅本地数据集），上传数据不能写入固定结果目录。
    if result_dir != "__UPLOADED__":
        save_pie_chart_background(bg_img, xlim, ylim, result_dir)
        
    return bg_img, (xlim, ylim)
