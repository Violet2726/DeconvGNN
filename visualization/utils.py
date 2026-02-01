
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from PIL import Image
import pandas as pd
import hashlib
from functools import lru_cache

from typing import Tuple, List, Optional, Any, Dict, Callable
from matplotlib.axes import Axes
import plotly.graph_objects as go

# Streamlit 缓存（延迟导入避免循环依赖）
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Top N 配置：只显示前 N 个最大的类别
TOP_N_CATEGORIES = 4

# ========== 性能优化配置 ==========
LOD_THRESHOLD = 5000  # 超过此点数启用 LOD 抽样
LOD_SAMPLE_RATIO = 0.3  # LOD 模式下的抽样比例
CHART_CACHE_SIZE = 16  # 图表缓存数量

# ========== 环境检测 ==========
def is_cloud_environment() -> bool:
    """
    检测是否在 Streamlit Cloud 环境中运行。
    用于在 UI 中显示/隐藏不适用于云端的功能。
    """
    import os
    # Streamlit Cloud 会设置这些环境变量
    return (
        os.environ.get("STREAMLIT_SHARING_MODE") is not None or
        os.environ.get("IS_STREAMLIT_CLOUD") is not None or
        os.path.exists("/mount/src")  # Streamlit Cloud 特有的挂载路径
    )

def get_data_fingerprint(df: pd.DataFrame) -> str:
    """
    生成 DataFrame 的快速指纹（用于缓存键）。
    使用形状 + 前后几行的哈希值，避免完整数据哈希的开销。
    """
    shape_str = f"{df.shape}"
    # 取前5行和后5行的字符串表示
    sample_str = df.head(5).to_string() + df.tail(5).to_string()
    return hashlib.md5((shape_str + sample_str).encode()).hexdigest()[:12]

# 图表缓存装饰器
def cached_chart(func):
    """
    Streamlit 兼容的图表缓存装饰器。
    在 Streamlit 环境下使用 st.cache_data，否则使用 lru_cache。
    """
    if HAS_STREAMLIT:
        return st.cache_data(ttl=1800, max_entries=CHART_CACHE_SIZE, show_spinner=False)(func)
    else:
        return lru_cache(maxsize=CHART_CACHE_SIZE)(func)


def get_adaptive_dpi(n_points: int) -> int:
    """
    根据数据量动态调整渲染 DPI，平衡质量与性能。
    
    Args:
        n_points: 数据点数量
    
    Returns:
        适合的 DPI 值
    """
    if n_points > 10000:
        return 150  # 大数据集用低 DPI，优先性能
    elif n_points > 5000:
        return 300  # 中等数据集
    elif n_points > 2000:
        return 450  # 较小数据集
    else:
        return 600  # 小数据集用高 DPI

def apply_lod_sampling(predict_df: pd.DataFrame, coords: pd.DataFrame, 
                       force_full: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    对大数据集应用 LOD（层次细节）抽样。
    
    Args:
        predict_df: 预测结果数据框
        coords: 坐标数据框
        force_full: 是否强制使用全部数据
    
    Returns:
        (抽样后的 predict_df, 抽样后的 coords, 是否进行了抽样)
    """
    n_points = len(predict_df)
    
    if force_full or n_points <= LOD_THRESHOLD:
        return predict_df, coords, False
    
    # 执行抽样
    sample_n = int(n_points * LOD_SAMPLE_RATIO)
    sample_indices = np.random.choice(predict_df.index, size=sample_n, replace=False)
    
    return predict_df.loc[sample_indices], coords.loc[sample_indices], True

def generate_clean_pie_chart(predict_df, coords, point_size=20, 
                              progress_callback: Optional[Callable[[float, str], None]] = None):
    """
    生成高分辨率、无边框的饼图背景（透明背景），用于叠加在 Plotly 图表下方。
    使用 matplotlib PatchCollection 优化大规模渲染性能。
    
    Args:
        predict_df: 预测结果数据框
        coords: 坐标数据框
        point_size: 点大小（None 为自动计算）
        progress_callback: 进度回调函数 (progress: 0-1, message: str)
    """
    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)
    
    update_progress(0.05, "准备颜色映射...")
    
    # 准备颜色 (使用智能聚类配色)
    labels = predict_df.columns.tolist()
    color_map = get_color_map(labels, predict_df)
    colors = [color_map[label] for label in labels]
    
    update_progress(0.1, "计算画布尺寸...")
    
    # 计算画布大小
    x_range = coords['x'].max() - coords['x'].min()
    y_range = coords['y'].max() - coords['y'].min()
    if y_range == 0: y_range = 1
    aspect_ratio = x_range / y_range
    
    # 设置自适应 DPI（性能优化关键）
    n_points = len(predict_df)
    dpi = get_adaptive_dpi(n_points)
    base_size = 12
    fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size), dpi=dpi)
    
    update_progress(0.15, f"使用 DPI={dpi} (数据点: {n_points})...")
    
    update_progress(0.2, "计算最佳点大小...")
    
    # --- 自动计算最佳点大小 ---
    from sklearn.neighbors import NearestNeighbors
    coords_array = np.column_stack((coords['x'], coords['y']))
    # 计算最近邻距离的中位数作为平均间距
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords_array)
    distances, _ = nbrs.kneighbors(coords_array)
    avg_spacing = np.median(distances[:, 1])
    
    update_progress(0.25, "初始化图形对象...")
    
    # 使用 PatchCollection 优化绘图性能
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection

    # 半径系数经验值
    radius = avg_spacing * 0.42
    
    print(f"  ℹ️ [Performance] 使用 PatchCollection 批量渲染优化 (DPI: {dpi}, 半径: {radius:.4f})")
    
    wedges = []
    
    # 预计算数据
    predict_values = predict_df.values
    x_coords = coords['x'].values
    y_coords = coords['y'].values
    
    # 全局 Top N
    top_n = TOP_N_CATEGORIES
    total_points = len(predict_df)
    
    update_progress(0.3, f"构建 {total_points} 个饼图对象...")

    for i in range(total_points):
        # 更新进度（每 10% 更新一次）
        if i % max(1, total_points // 10) == 0:
            progress = 0.3 + 0.5 * (i / total_points)
            update_progress(progress, f"处理点 {i}/{total_points}...")
        # 1. 获取当前点的分布和坐标
        dist = predict_values[i]
        xc, yc = x_coords[i], y_coords[i]
        
        # 2. 计算 Top N
        # 找出 Top N 的索引
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
        
        # 3. 归一化并构建扇形
        total = sum(current_vals)
        # 起始角度 (0度对应X轴正方向)
        current_angle = 0.0
        
        for val, color in zip(current_vals, current_colors):
            # 计算扇区跨度 (角度)
            ratio = val / total
            theta = ratio * 360.0
            
            # 创建扇形 Wedge((x,y), r, start, end)
            # 注意: Wedge 默认接受度数
            w = Wedge((xc, yc), radius, current_angle, current_angle + theta, facecolor=color, linewidth=0)
            wedges.append(w)
            
            current_angle += theta

    # 4. 一次性添加到 Axes
    if wedges:
        print(f"  ℹ️ [Rendering] 正在渲染 {len(wedges)} 个扇形图层...")
        # match_original=True 确保保留每个 Wedge 的颜色
        p = PatchCollection(wedges, match_original=True)
        ax.add_collection(p)
        
    # 设置坐标轴范围与 Plotly 严格一致
    # 增加一点点 padding 防止边缘被切
    padding = x_range * 0.05
    ax.set_xlim(coords['x'].min() - padding, coords['x'].max() + padding)
    ax.set_ylim(coords['y'].min() - padding, coords['y'].max() + padding)
    
    # 移除所有装饰
    ax.axis('off')
    plt.margins(0)
    # 确保完全无边距
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    
    # 设置透明背景
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    
    # 保存到 buffer
    update_progress(0.9, "保存图像...")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    update_progress(1.0, "完成！")
    return Image.open(buf), (ax.get_xlim(), ax.get_ylim())

def get_color_map(labels: List[str], predict_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    生成标签列表的颜色映射表。
    
    [智能配色模式]
    如果提供了 predict_df，则计算细胞类型间的空间相关性，并进行层次聚类。
    排序后的细胞类型将在色相环(Hue Cycle)上相邻，从而实现：
    "空间分布相似的细胞类型，颜色也相近"（符合学术界 co-localization 可视化标准）。
    
    [默认模式]
    如果未提供 predict_df，则按字母序排列。
    """
    import matplotlib.colors as mcolors
    import colorsys
    import numpy as np
    import pandas as pd
    
    # 默认按字母序，保证确定性
    sorted_labels = sorted(list(set(labels)))
    
    # --- 核心改进：基于生物学/空间相关性的层次聚类排序 ---
    if predict_df is not None:
        try:
            import scipy.cluster.hierarchy as sch
            import scipy.spatial.distance as dist
            
            # 1. 确保 predict_df 仅包含需要的列
            valid_cols = [c for c in sorted_labels if c in predict_df.columns]
            if len(valid_cols) > 2:
                data = predict_df[valid_cols]
                
                # 2. 计算特征矩阵的转置（行=细胞类型，列=空间点）
                # 我们要聚类的是"细胞类型"
                correlation_matrix = data.corr().fillna(0)
                
                # 3. 层次聚类 (Ward's method)
                # 使用 1 - correlation 作为距离度量
                d = sch.distance.pdist(correlation_matrix)
                linkage = sch.linkage(d, method='ward')
                
                # 4. 获取最优叶节点排序索引
                ind = sch.leaves_list(linkage)
                
                # 5. 重排标签
                sorted_labels = [correlation_matrix.columns[i] for i in ind]
                # print(f"  ℹ️ [Auto-Color] 已根据空间相关性重排细胞类型顺序")
        except Exception as e:
            print(f"  ⚠️ [Color] 聚类排序失败，回退到字母序: {e}")
            
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


def generate_plotly_scatter(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame, 
                          hover_count: int, bg_img: Any, bounds: Tuple[float, float], 
                          color_map: Dict[str, str]) -> go.Figure:
    """生成空间组成分布散点图 (Tab 1)，配合背景饼图使用。"""
    import plotly.express as px
    import plotly.graph_objects as go
    
    xlim, ylim = bounds
    plot_df = coords_for_plot.copy()
    
    # 优化悬停文本生成
    # 使用 numpy 进行批量处理，避免 pandas 的逐行操作
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
        title='空间组成分布'
    )
    
    fig.update_traces(
        marker=dict(opacity=0),
        hovertemplate='%{hovertext}<extra></extra>'
    )
    
    # 虚拟图例 - 添加所有细胞类型
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
    
    # 添加背景饼图图片
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
    """生成优势细胞类型散点图 (Tab 2)，自适应调整点大小。"""
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['主要细胞类型'] = predict_df.idxmax(axis=1).values
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    # Size calculation - 自适应大小
    # 启发式公式: 假设画布有效区域约600-800px, 避免重叠
    n_points = len(predict_df)
    adaptive_size = max(3, (550 / np.sqrt(n_points)))
    adaptive_size = min(15, adaptive_size) # 限制最大值
    
    display_df['pixel_size'] = adaptive_size
    
    print(f"  ℹ️ [Plotly] 自适应点大小: {adaptive_size:.2f} (N={n_points})")
        
    # 修改：不再强制字母排序，而是跟随 color_map 的键顺序（即聚类顺序）
    # unique_types = sorted(predict_df.columns.tolist())
    unique_types = [t for t in color_map.keys() if t in predict_df.columns]
    
    fig = go.Figure()
    
    for cell_type in unique_types:
        subset = display_df[display_df['主要细胞类型'] == cell_type]
        if len(subset) == 0: continue
            
        # 优化悬停文本生成
        # 我们需要将子集索引映射回 predict_df 中的原始整数位置
        # 或者更简单：为此子集重新计算（现在速度足够快）或使用严格查找
        
        # 在子集上使用高效处理方法
        subset_indices = subset.index
        subset_predict_vals = predict_df.loc[subset_indices].values
        subset_cols = predict_df.columns.tolist()
        
        # 批量排序获取 Top N
        sub_top_res = np.argsort(subset_predict_vals, axis=1)[:, -hover_count:][:, ::-1]
        
        hover_texts = []
        for i, idx_val in enumerate(subset_indices):
            major_type = cell_type # Known from loop
            major_prop = subset.loc[idx_val, '主要比例']
            
            text = f"<b>位置 {idx_val}</b><br>主要类型: {major_type} ({major_prop:.2%})<br>"
            
            row_indices = sub_top_res[i]
            for col_idx in row_indices:
                proportion = subset_predict_vals[i, col_idx]
                if proportion > 0:
                     ct = subset_cols[col_idx]
                     text += f"{ct}: {proportion:.2%}<br>"
            hover_texts.append(text)

        fig.add_trace(go.Scatter(
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
    """为 Tab 3 生成柱状图"""
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
    # 根据条目数量动态调整高度，防止拥挤
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
    """为 Tab 4 生成热图"""
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['proportion'] = predict_df[selected_type].values
    
    hover_texts = [f"<b>位置 {idx}</b><br>类型: {selected_type}<br>比例: {val:.2%}" 
                  for idx, val in zip(display_df.index, display_df['proportion'])]

    
    # 自适应大小
    n_points = len(predict_df)
    adaptive_size = max(3, (550 / np.sqrt(n_points)))
    adaptive_size = min(15, adaptive_size)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
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
    """保存生成的饼图背景及元数据 (xlim/ylim)。"""
    import os
    import json
    
    target_img = os.path.join(result_dir, "interactive_pie_background.png")
    target_meta = os.path.join(result_dir, "interactive_pie_bounds.json")
    
    try:
         img.save(target_img)
         with open(target_meta, 'w') as f:
             json.dump({'xlim': xlim, 'ylim': ylim}, f)
    except Exception as e:
         print(f"警告: 无法保存背景缓存: {e}")

def open_folder_dialog() -> Optional[str]:
    """弹出系统文件夹选择框 (仅本地运行有效)。"""
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
    except Exception as e:
        print(f"打开文件夹对话框出错: {e}")
        return None

def generate_and_save_interactive_assets(predict_df, coordinates, output_dir):
    """
    生成并保存交互式可视化资源 (Top 4 饼图背景)。
    自动处理坐标映射并保存为 interactive_pie_background.png。
    """
    import os
    
    print(f"正在生成交互式可视化背景图 (Top {TOP_N_CATEGORIES})...")
    
    # 确保坐标列名匹配
    # generate_clean_pie_chart_top_n 期望 coordinates 有 'x', 'y' 列
    coords = coordinates.copy()
    if 'coor_X' in coords.columns:
        coords = coords.rename(columns={'coor_X': 'x', 'coor_Y': 'y'})
    elif 'x' not in coords.columns:
        # 假设前两列是 x, y
        coords.columns = ['x', 'y']
        
    # 确保索引对齐
    common_index = predict_df.index.intersection(coords.index)
    if len(common_index) < len(predict_df):
        print(f"警告: 坐标与预测结果索引不完全匹配。交集: {len(common_index)}")
    
    predict_df = predict_df.loc[common_index]
    coords = coords.loc[common_index]

    # 生成图片
    # point_size=None 让其自动计算
    try:
        img, (xlim, ylim) = generate_clean_pie_chart(predict_df, coords, point_size=None)
        
        # 保存
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        save_pie_chart_background(img, xlim, ylim, output_dir)
        print(f"交互式背景图已保存至: {output_dir}")
    except Exception as e:
        print(f"生成可视化资源时出错: {e}")

def handle_visualization_generation(paths):
    """
    可视化生成流程的单一入口函数：读取数据 -> 生成背景 -> 保存资源。
    """
    import os
    import pandas as pd
    
    print("\n" + "="*60)
    print("[Visualization] 生成交互式可视化资源 (Top 4 饼图)...")
    print("="*60)
    
    res_path = os.path.join(paths['output_path'], 'predict_result.csv')
    coor_path = os.path.join(paths['ST_path'], 'coordinates.csv')
    
    if os.path.exists(res_path) and os.path.exists(coor_path):
        try:
            # 读取预测结果和坐标
            pred_df = pd.read_csv(res_path, index_col=0)
            # 处理坐标文件读取，有的包含表头有的不包含，这里假设标准格式
            coord_df = pd.read_csv(coor_path, index_col=0)
            
            # 调用生成函数
            generate_and_save_interactive_assets(pred_df, coord_df, paths['output_path'])
            print("[SUCCESS] 可视化资源生成完成！")
        except Exception as e:
            print(f"[ERROR] 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[WARN] 找不到结果文件或坐标文件，跳过可视化生成。")
        print(f"  预测结果: {res_path} ({'存在' if os.path.exists(res_path) else '缺失'})")
        print(f"  坐标文件: {coor_path} ({'存在' if os.path.exists(coor_path) else '缺失'})")
