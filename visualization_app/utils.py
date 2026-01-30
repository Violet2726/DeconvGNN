
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

# Top N 配置：只显示前 N 个最大的类别
TOP_N_CATEGORIES = 4

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
    
    # 使用 PatchCollection 优化绘图性能
    # 相比于循环调用 scatter，这将大幅减少渲染开销
    from matplotlib.patches import Wedge
    from matplotlib.collections import PatchCollection

    # 直接使用数据单位的半径 (之前已经利用 s 计算过了，这里取回原始逻辑)
    # radius_unit = avg_spacing * 0.42
    # 我们重新定义一下半径，确保与之前的逻辑一致
    # 之前: radius_unit = avg_spacing * 0.42
    radius = avg_spacing * 0.42
    
    print(f"  ℹ️ [Performance] 使用 PatchCollection 批量渲染优化 (半径: {radius:.4f})")
    
    wedges = []
    
    # 预计算数据
    predict_values = predict_df.values
    x_coords = coords['x'].values
    y_coords = coords['y'].values
    
    # 引入 tqdm 显示进度
    try:
        from tqdm import tqdm
        iterator = tqdm(range(len(predict_df)), desc="构建图形对象", unit="spot")
    except ImportError:
        iterator = range(len(predict_df))

    # 全局 Top N
    top_n = TOP_N_CATEGORIES

    for i in iterator:
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
    
    # Optimize Hover text generation
    # Use numpy for batch processing instead of pandas row-wise operations
    vals = predict_df.values
    cols = predict_df.columns.tolist()
    
    # Get indices of top N elements for each row
    # np.argsort sorts ascending, so we take the last hover_count elements and reverse them
    top_n_indices = np.argsort(vals, axis=1)[:, -hover_count:][:, ::-1]
    
    hover_texts = []
    indices = predict_df.index.tolist()
    
    for i in range(len(predict_df)):
        idx_label = indices[i]
        text = f"<b>位置 {idx_label}</b><br>"
        
        # Determine strict Top N for this row
        row_indices = top_n_indices[i]
        
        for col_idx in row_indices:
            proportion = vals[i, col_idx]
            if proportion > 0: # Only show non-zero
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
    
    # Dummy legend - 添加所有细胞类型
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
    
    display_df['主要比例'] = predict_df.max(axis=1).values
    
    # Size calculation - 自适应大小
    # 启发式公式: 假设画布有效区域约600-800px, 避免重叠
    n_points = len(predict_df)
    adaptive_size = max(3, (550 / np.sqrt(n_points)))
    adaptive_size = min(15, adaptive_size) # 限制最大值
    
    display_df['pixel_size'] = adaptive_size
    
    print(f"  ℹ️ [Plotly] 自适应点大小: {adaptive_size:.2f} (N={n_points})")
        
    unique_types = sorted(predict_df.columns.tolist())
    fig = go.Figure()
    
    for cell_type in unique_types:
        subset = display_df[display_df['主要细胞类型'] == cell_type]
        if len(subset) == 0: continue
            
        # Optimize Hover text generation using the pre-calculated global indices
        # We need to map the subset indices back to the original integer location in predict_df
        # Or simpler: re-calculate for this subset (it's fast enough now) or use strict lookups.
        
        # Let's use the efficient approach on the subset
        subset_indices = subset.index
        subset_predict_vals = predict_df.loc[subset_indices].values
        subset_cols = predict_df.columns.tolist()
        
        # Batch sort for top N
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

def save_pie_chart_background(img: Image.Image, xlim: float, ylim: float, result_dir: str) -> None:
    """
    Save the generated pie chart background and its metadata to the result directory.
    
    Args:
        img (Image.Image): The generated background image.
        xlim (float): The limit of the x-axis.
        ylim (float): The limit of the y-axis.
        result_dir (str): The directory to save the files in.
    """
    import os
    import json
    
    target_img = os.path.join(result_dir, "interactive_pie_background.png")
    target_meta = os.path.join(result_dir, "interactive_pie_bounds.json")
    
    try:
         img.save(target_img)
         with open(target_meta, 'w') as f:
             json.dump({'xlim': xlim, 'ylim': ylim}, f)
    except Exception as e:
         print(f"Warning: Failed to save background cache: {e}")

def open_folder_dialog() -> Optional[str]:
    """
    Open a system folder selection dialog using tkinter.
    
    Returns:
        Optional[str]: The selected folder path, or None if cancelled/failed.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        # Set up root window, hide it, and make it top-most
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Open dialog
        folder = filedialog.askdirectory(title="选择 STdGCN 输出目录")
        
        # Clean up
        root.destroy()
        return folder if folder else None
    except Exception as e:
        print(f"Error opening folder dialog: {e}")
        return None

def generate_and_save_interactive_assets(predict_df, coordinates, output_dir):
    """
    生成并保存交互式可视化所需的背景图和元数据。
    调用 generate_clean_pie_chart_top_n 生成 Top 4 饼图背景。
    供 Tutorial.py 等外部脚本调用。
    
    Args:
        predict_df: 预测结果 DataFrame (index=Barcodes, columns=Cell Types)
        coordinates: 坐标 DataFrame (index=Barcodes, columns=['X', 'Y'])
        output_dir: 结果输出目录
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
        print(f"Warning: 坐标与预测结果索引不完全匹配。交集: {len(common_index)}")
    
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
        print(f"Error generating visualization: {e}")

def handle_visualization_generation(paths):
    """
    处理可视化的完整流程:
    1. 检查预测结果和坐标文件是否存在
    2. 读取数据
    3. 调用 generate_and_save_interactive_assets 生成资源
    
    Args:
        paths: 包含 'output_path' 和 'ST_path' 的字典
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
