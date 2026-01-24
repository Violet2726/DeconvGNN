
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from PIL import Image

def draw_pie(dist, xpos, ypos, size, colors, ax):
    """绘制单个饼图"""
    cumsum = np.cumsum(dist)
    cumsum = cumsum / cumsum[-1]
    pie = [0] + cumsum.tolist()
    i = 0
    for r1, r2 in zip(pie[:-1], pie[1:]):
        angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2, num=30)
        x = [0] + np.cos(angles).tolist()
        y = [0] + np.sin(angles).tolist()

        xy = np.column_stack([x, y])
        ax.scatter([xpos], [ypos], marker=xy, s=size, c=colors[i], edgecolors='none')
        i += 1
    return ax

def generate_clean_pie_chart(predict_df, coords, point_size=20):
    """
    生成纯净的饼图背景图片（无坐标轴、无白边）
    返回: PIL Image 对象
    """
    # 准备颜色
    labels = predict_df.columns.tolist()
    if len(labels) <= 10:
        colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
    else:
        import matplotlib
        color = plt.get_cmap('rainbow', len(labels))
        colors = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in color(range(len(labels)))]
    
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
