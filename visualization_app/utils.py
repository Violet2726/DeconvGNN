
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from PIL import Image
import pandas as pd

from typing import Tuple, List, Optional, Any, Dict
from typing import Tuple, List, Optional, Any, Dict
from matplotlib.axes import Axes
import plotly.graph_objects as go

def draw_pie(dist: np.ndarray, xpos: float, ypos: float, size: float, 
             colors: List[str], ax: Axes) -> Axes:
    """
    在 Matplotlib Axes 上绘制单个散点饼图。

    Args:
        dist (np.ndarray): 细胞类型比例分布数组。
        xpos (float): x 坐标。
        ypos (float): y 坐标。
        size (float): 点的大小 (area).
        colors (List[str]): 颜色列表。
        ax (Axes): Matplotlib Axes 对象。

    Returns:
        Axes: 更新后的 Axes 对象。
    """
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0.0] + cumsum.tolist()
    
    for i, (r1, r2) in enumerate(zip(pie[:-1], pie[1:])):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=30)
        x = [0.0] + np.cos(angles).tolist()
        y = [0.0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, c=colors[i], edgecolors='none')
        
    return ax

def generate_clean_pie_chart(predict_df, coords, point_size=20):
    """
    生成纯净的饼图背景图片（无坐标轴、无白边）
    返回: PIL Image 对象
    """
    # 准备颜色
    labels = predict_df.columns.tolist()
    color_map = get_color_map(labels)
    colors = [color_map[label] for label in labels]
    
    # 计算画布大小
    x_range = coords['x'].max() - coords['x'].min()
    y_range = coords['y'].max() - coords['y'].min()
    if y_range == 0: y_range = 1  # 防止除以零
    aspect_ratio = x_range / y_range
    
    # 设置高分辨率画布 (提高DPI)
    dpi = 200
    base_size = 12
    fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size), dpi=dpi)
    
    # --- 自动计算最佳点大小 ---
    from sklearn.neighbors import NearestNeighbors
    coords_array = np.column_stack((coords['x'], coords['y']))
    # 只需要计算最近的一个邻居
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords_array)
    distances, _ = nbrs.kneighbors(coords_array)
    # 取最近邻距离的中位数作为平均间距
    avg_spacing = np.median(distances[:, 1])
    
    # 理想情况下，点的大小应略小于间距，防止重叠
    # Matplotlib 的 scatter s 参数也是面积，关系比较复杂
    # 经过经验调整的公式：将数据坐标系下的"半径"转换为图形坐标系下的"点大小"
    
    # 1. 计算 ax 的总像素宽度
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_px = bbox.width * dpi
    
    # 2. 计算每个数据单位对应的像素数
    px_per_unit = width_px / x_range
    
    # 3. 设定点的半径为平均间距的 0.45 倍 (直径0.9，留一点缝隙)
    radius_unit = avg_spacing * 0.42
    
    # 4. 转换为 s 参数 (area in points^2)
    # s = (radius_in_points * 2) ^ 2
    # 1 inch = 72 points
    radius_px = radius_unit * px_per_unit
    radius_points = radius_px * (72 / dpi)
    calculated_s = (radius_points * 2) ** 2
    
    # 允许手动覆盖，或者使用自动计算值
    final_size = calculated_s if point_size is None or point_size == 20 else point_size
    
    print(f"  ℹ️ [Auto-Size] 平均间距: {avg_spacing:.4f}, 计算点大小(s): {calculated_s:.2f}")

    # 预计算数据
    predict_values = predict_df.values
    x_coords = coords['x'].values
    y_coords = coords['y'].values
    
    for i in range(len(predict_df)):
        draw_pie(predict_values[i], x_coords[i], y_coords[i], final_size, colors, ax)
        
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
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    return Image.open(buf), (ax.get_xlim(), ax.get_ylim())

def get_color_map(labels: List[str]) -> Dict[str, str]:
    """
    Generate a color map for a list of labels (e.g. cell types).
    
    Args:
        labels (List[str]): List of labels.
        
    Returns:
        Dict[str, str]: Dictionary mapping labels to hex color codes.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    
    unique_types = sorted(list(set(labels)))
    if len(unique_types) <= 10:
        colors_list = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(unique_types)]
    else:
        color_tab = plt.get_cmap('rainbow', len(unique_types))
        colors_list = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in color_tab(range(len(unique_types)))]
    
    return dict(zip(unique_types, colors_list))

def generate_plotly_scatter(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame, 
                          hover_count: int, bg_img: Any, bounds: Tuple[float, float], 
                          color_map: Dict[str, str]) -> go.Figure:
    """
    Generate the interactive Plotly scatter plot for Tab 1 (Spatial Composition).
    """
    import plotly.express as px
    import plotly.graph_objects as go
    
    xlim, ylim = bounds
    plot_df = coords_for_plot.copy()
    
    # Hover text
    hover_texts = []
    for idx in range(len(predict_df)):
        row = predict_df.iloc[idx]
        sorted_row = row.sort_values(ascending=False)
        text = f"<b>位置 {predict_df.index[idx]}</b><br>"
        for cell_type, proportion in sorted_row.head(hover_count).items():
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
    
    # Dummy legend
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
    
    # Background image
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
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            title="细胞类型 (饼图颜色)",
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            itemclick=False, itemdoubleclick=False
        ),
        dragmode='pan'
    )
    return fig

def generate_dominant_scatter(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame,
                             hover_count: int, color_map: Dict[str, str]) -> go.Figure:
    """
    Generate the dominant cell type scatter plot for Tab 2.
    """
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['主要细胞类型'] = predict_df.idxmax(axis=1).values
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    # Size calculation
    p = display_df['主要比例'].values
    if len(p) > 0:
        min_p, max_p = p.min(), p.max()
        normalized = (p - min_p) / (max_p - min_p + 1e-6)
        exp_normalized = (np.exp(2.0 * normalized) - 1) / (np.exp(2.0) - 1)
        display_df['pixel_size'] = 8 + exp_normalized * 6
    else:
        display_df['pixel_size'] = 10
        
    unique_types = sorted(predict_df.columns.tolist())
    fig = go.Figure()
    
    for cell_type in unique_types:
        subset = display_df[display_df['主要细胞类型'] == cell_type]
        if len(subset) == 0: continue
            
        hover_texts = []
        for idx in subset.index:
            row = predict_df.loc[idx]
            sorted_row = row.sort_values(ascending=False)
            text = f"<b>位置 {idx}</b><br>主要类型: {cell_type} ({subset.loc[idx, '主要比例']:.2%})<br>"
            for ct, prop in sorted_row.head(hover_count).items():
                text += f"{ct}: {prop:.2%}<br>"
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
        height=800, autosize=True,title='主要类型分布',
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, showgrid=False),
        xaxis=dict(visible=False, showgrid=False),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        dragmode='pan',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=0)
    )
    return fig

def generate_proportion_bar(predict_df: pd.DataFrame) -> go.Figure:
    """Generate bar chart for Tab 3"""
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
    fig.update_layout(height=500, showlegend=False)
    return fig

def generate_heatmap(coords_for_plot: pd.DataFrame, predict_df: pd.DataFrame, 
                    selected_type: str) -> go.Figure:
    """Generate heatmap for Tab 4"""
    import plotly.graph_objects as go
    
    display_df = coords_for_plot.copy()
    display_df['proportion'] = predict_df[selected_type].values
    
    hover_texts = [f"<b>位置 {idx}</b><br>类型: {selected_type}<br>比例: {val:.2%}" 
                  for idx, val in zip(display_df.index, display_df['proportion'])]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df['x'], y=display_df['y'],
        mode='markers',
        marker=dict(
            size=12,
            color=display_df['proportion'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="比例"),
            opacity=1.0
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        height=800, autosize=True,
        yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, showgrid=False),
        xaxis=dict(visible=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        dragmode='pan', margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig
