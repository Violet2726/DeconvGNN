import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm.auto import tqdm



def draw_pie(dist, xpos, ypos, size, colors, ax):

    cumsum = np.cumsum(dist)
    cumsum = cumsum/ cumsum[-1]
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



def plot_frac_results(predict, cell_type_list, coordinates, file_name=None, point_size=None, size_coefficient=0.0009, if_show=True, color_dict=None):
    """
    绘制空间组成分布饼图 (优化版)
    支持自动计算点大小、高清渲染
    """
    from sklearn.neighbors import NearestNeighbors
    
    coordinates.columns = ['coor_X', 'coor_Y']
    labels = cell_type_list
    
    # 颜色映射处理
    if color_dict is not None:
        colors = [color_dict[i] for i in cell_type_list]
    else:
        if len(labels) <= 10:
            colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
        else:
            import matplotlib
            # 修正原有的颜色列表生成逻辑
            cmap = plt.get_cmap('rainbow', len(labels))
            colors = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in cmap(range(len(labels)))]
    
    # 计算画布大小
    x_min, x_max = coordinates['coor_X'].min(), coordinates['coor_X'].max()
    y_min, y_max = coordinates['coor_Y'].min(), coordinates['coor_Y'].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    if y_range == 0: y_range = 1
    aspect_ratio = x_range / y_range
    
    # 预留右侧给图例的空间
    str_len = max([len(item) for item in cell_type_list]) if len(cell_type_list) > 0 else 0
    extend_region_ratio = 1.3  # 简单的比例扩充
    
    dpi = 300
    base_size = 10
    fig, ax = plt.subplots(figsize=(base_size * aspect_ratio * extend_region_ratio, base_size), dpi=dpi)
    
    # --- 自动计算最佳点大小 (如果 point_size 为 None 或 默认值) ---
    # 注意：STdGCN 默认传参是 300 或 1000，这里我们将其视为"未指定"如果它看起来不合理，
    # 或者我们显式地计算一个参考值并打印出来供参考。
    # 为了完全复用新逻辑，我们优先使用自动计算。
    
    coords_array = np.column_stack((coordinates['coor_X'], coordinates['coor_Y']))
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords_array)
    distances, _ = nbrs.kneighbors(coords_array)
    avg_spacing = np.median(distances[:, 1])
    
    # 转换逻辑
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_px = bbox.width * dpi
    # 此时 ax 的 xlim 尚未设定，我们需要先假定一个范围
    # 为了计算准确，我们需要先设定初步的 lim
    # 增加 5% padding 以防止边缘的点被切断
    padding_x = x_range * 0.05
    padding_y = y_range * 0.05
    ax.set_xlim(x_min - padding_x, x_max + padding_x)
    ax.set_ylim(y_min - padding_y, y_max + padding_y)
    
    # 重新获取转换比例
    # 注意: tight_layout 或 legend 可能会改变 ax 大小
    # 我们使用更稳健的半径计算法:
    
    # 理想半径 (数据单位)
    radius_unit = avg_spacing * 0.45 
    
    # 转换系数: 数据单位 -> 点 (Points)
    # s = (radius_in_points * 2) ** 2
    # 我们需要获取 ax 的 transform
    # 这里为了简化，我们使用一个相对鲁棒的估计
    
    # 简单策略：如果用户没指定(None)或者指定的是旧的默认大值(>=300)，我们使用自动计算
    if point_size is None or point_size >= 100:
        # 重新计算 transform
        # Figure 宽度 (inches) * dpi = 像素宽度
        # Data Range X
        px_per_unit = (base_size * aspect_ratio * dpi) / (x_range * 1.5) # 1.5 是考虑图例占位
        
        radius_px = radius_unit * px_per_unit
        radius_points = radius_px * (72 / dpi)
        calculated_s = (radius_points * 2) ** 2
        
        final_size = calculated_s
        print(f"  ℹ️ [Auto-Size] 计算点大小(s): {final_size:.2f} (原参数: {point_size})")
    else:
        final_size = point_size

    # 绘制
    for i in tqdm(range(predict.shape[0]), desc="Plotting pie plots:"):
        ax = draw_pie(predict[i], coordinates['coor_X'].values[i], coordinates['coor_Y'].values[i], 
                      size=final_size, ax=ax, colors=colors)
    
    # 图例
    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(len(colors))]
    # 动态调整字体大小
    fontsize = 12
    ax.legend(handles=patches, fontsize=fontsize, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)
    
    ax.axis("off")
    # 保持比例
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if file_name is not None:
        plt.savefig(file_name, dpi=dpi, bbox_inches='tight')
        
    if if_show:
        plt.show()
    plt.close('all')


def save_interactive_assets(predict, cell_type_list, coordinates, output_dir):
    """
    生成 Web 端交互所需的纯净背景图和坐标元数据
    1. interactive_pie_background.png
    2. interactive_pie_bounds.json
    """
    import json
    import os
    from sklearn.neighbors import NearestNeighbors
    
    # 确保列名正确
    coordinates.columns = ['coor_X', 'coor_Y']
    labels = cell_type_list
    
    # 颜色生成 (与 plot_frac_results 保持一致)
    if len(labels) <= 10:
        colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
    else:
        import matplotlib
        cmap = plt.get_cmap('rainbow', len(labels))
        colors = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in cmap(range(len(labels)))]
    
    # 计算原始范围
    x_min, x_max = coordinates['coor_X'].min(), coordinates['coor_X'].max()
    y_min, y_max = coordinates['coor_Y'].min(), coordinates['coor_Y'].max()

    # 1. 先计算点的大小半径 (数据单位)
    coords_array = np.column_stack((coordinates['coor_X'], coordinates['coor_Y']))
    nbrs = NearestNeighbors(n_neighbors=2).fit(coords_array)
    distances, _ = nbrs.kneighbors(coords_array)
    avg_spacing = np.median(distances[:, 1])
    
    # 使用针对 Web 优化的半径系数 (稍小一点，避免密集时视觉粘连)
    radius_unit = avg_spacing * 0.42
    
    # 2. 根据物理半径计算 padding
    # 确保边缘的点完全包含在视野内，使用 1.5 倍半径作为 buffer
    padding_val = radius_unit * 1.5
    
    final_xlim = (x_min - padding_val, x_max + padding_val)
    final_ylim = (y_min - padding_val, y_max + padding_val)
    
    # 3. 重新计算 Aspect Ratio 以匹配 padding 后的范围
    x_range = final_xlim[1] - final_xlim[0]
    y_range = final_ylim[1] - final_ylim[0]
    if y_range == 0: y_range = 1
    aspect_ratio = x_range / y_range
    
    dpi = 200 # Web 端不需要 300dpi，200够用且加载快
    base_size = 12
    # 纯净模式：不需要 extend_region_ratio
    fig, ax = plt.subplots(figsize=(base_size * aspect_ratio, base_size), dpi=dpi)
    
    ax.set_xlim(final_xlim)
    ax.set_ylim(final_ylim)
    
    # 计算像素点大小
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_px = bbox.width * dpi
    current_x_range = final_xlim[1] - final_xlim[0]
    px_per_unit = width_px / current_x_range
    
    radius_px = radius_unit * px_per_unit
    radius_points = radius_px * (72 / dpi)
    final_size = (radius_points * 2) ** 2
    
    print(f"  generating interactive assets... (point_size={final_size:.2f})")

    # 绘制
    for i in tqdm(range(predict.shape[0]), desc="Generating Web Assets"):
        ax = draw_pie(predict[i], coordinates['coor_X'].values[i], coordinates['coor_Y'].values[i], 
                      size=final_size, ax=ax, colors=colors)
    
    # 纯净设置
    ax.axis("off")
    plt.margins(0)
    
    # 1. 保存图片
    img_path = os.path.join(output_dir, "interactive_pie_background.png")
    
    # 确保 Axes 填满整个 Figure，避免白边或比例失调
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # 使用 transparent=True 确保背景透明，移除 bbox_inches='tight' 以保证坐标与 JSON 严格对应
    plt.savefig(img_path, dpi=dpi, transparent=True)
    plt.close('all')
    
    # 2. 保存元数据 (JSON)
    # 注意：bbox_inches='tight' 可能会微调实际的保存范围，导致 Plotly 对齐会有微小偏差。
    # 为了完美对齐，Web 端通常依赖图片本身的宽高比和这里记录的 Data Limits。
    # 我们记录刚才 set_xlim/ylim 的值。
    metadata = {
        "xlim": final_xlim,
        "ylim": final_ylim
    }
    json_path = os.path.join(output_dir, "interactive_pie_bounds.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f)
        
    print(f"  Web assets saved to: {output_dir}")
    
    
    
def plot_scatter_by_type(predict, cell_type_list, coordinates, point_size=400, size_coefficient=0.0009, file_path=None, if_show=True):
    
    coordinates.columns = ['coor_X', 'coor_Y']
    
    for i in tqdm(range(len(cell_type_list)), desc="Plotting cell type scatter plot:"):
        
        fig, ax = plt.subplots(figsize=(len(coordinates['coor_X'].unique())*point_size*size_coefficient+1, len(coordinates['coor_Y'].unique())*point_size*size_coefficient))
        cm = plt.cm.get_cmap('Reds')
        ax = plt.scatter(coordinates['coor_X'], coordinates['coor_Y'], s=point_size, vmin=0, vmax=1, c=predict[:, i], cmap=cm)

        cbar = plt.colorbar(ax, fraction=0.05)
        labelsize = max(predict.shape[0]/100, 10)
        labelsize = min(labelsize, 30)
        cbar.ax.tick_params(labelsize=labelsize)
        plt.axis("equal")
        plt.xticks([])
        plt.yticks([])
        plt.xlim(coordinates['coor_X'].min()-0.5, coordinates['coor_X'].max()+0.5)
        plt.ylim(coordinates['coor_Y'].min()-0.5, coordinates['coor_Y'].max()+0.5)
        plt.tight_layout()
        if file_path != None:
            name = cell_type_list[i].replace('/', '_')
            plt.savefig(file_path+'/{}.jpg'.format(name), dpi=300, bbox_inches='tight')
        if if_show == True:
            plt.show()
        plt.close('all')