"""
STdGCN 空间转录组反卷积可视化系统
主应用入口
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# 页面配置
st.set_page_config(
    page_title="STdGCN 可视化系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    /* 隐藏 Streamlit 默认菜单和水印，保留侧边栏按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 数据目录配置
DATA_DIRS = {
    "Visium 小鼠大脑 (2695 spots)": "output/visium_results",
    "seqFISH+ 真实数据 (72 spots)": "output/seqfish_results",
    "STARmap 模拟数据 (189 spots)": "output/stdgcn_starmap",
}

@st.cache_data
def load_results(result_dir):
    """加载反卷积结果"""
    predict_path = os.path.join(result_dir, "predict_result.csv")
    if not os.path.exists(predict_path):
        return None, None
    
    predict_df = pd.read_csv(predict_path, index_col=0)
    
    # 尝试加载坐标
    coords = None
    for data_dir in ["data/visium_combined", "data/seqfish_tsv", "data/starmap_tsv"]:
        coord_path = os.path.join(data_dir, "coordinates.csv")
        if os.path.exists(coord_path):
            try:
                coords = pd.read_csv(coord_path, index_col=0)
                if len(coords) == len(predict_df):
                    break
            except:
                continue
    
    return predict_df, coords

def get_cell_types(predict_df):
    """获取细胞类型列表"""
    return predict_df.columns.tolist()

def main():
    # 标题
    st.markdown('<p class="main-header">🧬 STdGCN 空间转录组反卷积可视化系统</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于图神经网络的细胞类型反卷积结果展示</p>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("📊 数据选择")
        
        # 数据集选择
        dataset = st.selectbox(
            "选择数据集",
            list(DATA_DIRS.keys()),
            index=0
        )
        result_dir = DATA_DIRS[dataset]
        
        st.divider()
        
        # 加载数据
        predict_df, coords = load_results(result_dir)
        
        if predict_df is not None:
            cell_types = get_cell_types(predict_df)
            
            # 侧边栏不再显示具体设置，保持整洁
            pass
        else:
            st.error("❌ 未找到结果文件")
            st.info(f"请先运行 Tutorial.py 生成结果")
            return
    
    # 主内容区
    if predict_df is not None:
        # 第一行：统计信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("空间点数量", len(predict_df))
        with col2:
            st.metric("细胞类型数", len(cell_types))
        with col3:
            st.metric("主要细胞类型", predict_df.mean().idxmax())
        with col4:
            st.metric("平均比例", f"{predict_df[predict_df.mean().idxmax()].mean():.2%}")
        
        st.divider()
        
        # 第二行：可视化标签页
        # 第二行：可视化标签页
        tabs = st.tabs([
            "🎨 空间组成分布", 
            "🔍 主要类型分布", 
            "📊 整体比例统计", 
            "🔥 单细胞类型热图", 
            "📈 模型训练监控", 
            "� 详细数据表"
        ])
        
        # --- Tab 1: 空间组成分布 (原交互式饼图模式) ---
        with tabs[0]:
            st.subheader("空间组成分布 (多色饼图)")
            
            from visualization_app.utils import generate_clean_pie_chart
            
            # 检查坐标数据
            coords_for_plot = None
            for data_dir in ["data/visium_combined", "data/seqfish_tsv", "data/starmap_tsv"]:
                coord_path = os.path.join(data_dir, "coordinates.csv")
                if os.path.exists(coord_path):
                    try:
                        temp_coords = pd.read_csv(coord_path, index_col=0)
                        if len(temp_coords) == len(predict_df):
                            coords_for_plot = temp_coords
                            break
                    except:
                        continue
            
            if coords_for_plot is not None:
                # 1. 尝试加载预生成的背景图
                bg_img = None
                xlim, ylim = None, None
                
                precomputed_img_path = os.path.join(result_dir, "interactive_pie_background.png")
                precomputed_meta_path = os.path.join(result_dir, "interactive_pie_bounds.json")
                
                if os.path.exists(precomputed_img_path) and os.path.exists(precomputed_meta_path):
                    from PIL import Image
                    import json
                    bg_img = Image.open(precomputed_img_path)
                    with open(precomputed_meta_path, 'r') as f:
                        metadata = json.load(f)
                        xlim = metadata['xlim']
                        ylim = metadata['ylim']
                    st.caption("✅ 已加载预生成的高清背景图")
                else:
                    st.info("💡 正在实时生成背景图（建议运行 generate_all_pie_charts.py 提前生成以加速）...")
                    with st.spinner("⏳ 正在绘制饼图背景..."):
                        @st.cache_data(persist=True, show_spinner=False)
                        def get_cached_background(df, cds, size):
                            from visualization_app.utils import generate_clean_pie_chart
                            return generate_clean_pie_chart(df, cds, size)
                        
                        bg_img, (xlim, ylim) = get_cached_background(predict_df, coords_for_plot, None)
                
                # 2. 准备交互数据（透明散点）
                import plotly.express as px
                import plotly.graph_objects as go
            
                plot_df = coords_for_plot.copy()
                
                # 构建悬停文本
                hover_texts = []
                for idx in range(len(predict_df)):
                    row = predict_df.iloc[idx]
                    sorted_row = row.sort_values(ascending=False)
                    text = f"<b>位置 {predict_df.index[idx]}</b><br>"
                    for cell_type, proportion in sorted_row.head(6).items(): # 默认显示前6个
                        bar = "█" * int(proportion * 20)
                        text += f"{cell_type}: {proportion:.2%}<br>"
                    hover_texts.append(text)
                plot_df['hover_text'] = hover_texts
                
                # 3. 准备颜色映射（与饼图生成的逻辑保持一致）
                import matplotlib.pyplot as plt
                import matplotlib
                labels = predict_df.columns.tolist()
                if len(labels) <= 10:
                    colors = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(labels)]
                else:
                    color_map = plt.get_cmap('rainbow', len(labels))
                    colors = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in color_map(range(len(labels)))]
                
                cell_type_color_map = dict(zip(labels, colors))

                # 4. 创建 Plotly 图表
                fig = px.scatter(
                    plot_df, x='x', y='y',
                    hover_name='hover_text',
                    title='空间组成分布'
                )
                
                # 设置点完全透明（作为交互层）
                fig.update_traces(
                    marker=dict(opacity=0),
                    hovertemplate='%{hovertext}<extra></extra>'
                )
                
                # 5. 添加"虚拟"图例 (纯展示)
                for cell_type, color in cell_type_color_map.items():
                    fig.add_trace(
                        go.Scatter(
                            x=[None], y=[None],
                            mode='markers',
                            marker=dict(size=10, color=color, symbol='circle'),
                            name=cell_type,
                            showlegend=True
                        )
                    )
                
                # 6. 添加背景图片
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
                
                # 7. 坐标轴设置
                fig.update_xaxes(range=[xlim[0], xlim[1]], visible=False, showgrid=False)
                fig.update_yaxes(range=[ylim[0], ylim[1]], visible=False, showgrid=False, scaleanchor="x", scaleratio=1)
                
                fig.update_layout(
                    height=650,
                    margin=dict(l=0, r=0, t=30, b=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        title="细胞类型 (饼图颜色)",
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02,
                        itemclick=False,
                        itemdoubleclick=False
                    ),
                    dragmode='pan'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 说明：此图背景为多色饼图，展示每个位置的细胞组成；鼠标悬停可查看具体比例数据。")
            else:
                 st.warning("缺少坐标数据，无法生成交互式图表。显示静态预览：")
                 pie_plot_path = os.path.join(result_dir, "predict_results_pie_plot.jpg")
                 st.image(pie_plot_path, use_container_width=True)

        # --- Tab 2: 主要类型分布 (原交互式散点模式) ---
        with tabs[1]:
            st.subheader("主要类型分布 (优势细胞)")
            
            # 控件区域
            col_ctrl1, col_ctrl2 = st.columns([1, 1])
            with col_ctrl1:
                st.markdown("##### ⚙️ 显示设置")
                hover_count = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)), key="tab2_hover")
            with col_ctrl2:
                st.markdown("##### 👁️ 图例控制")
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    select_all = st.button("全选", use_container_width=True, key="tab2_all")
                with col_btn2:
                    deselect_all = st.button("全不选", use_container_width=True, key="tab2_none")
            
            # 重新加载或复用坐标数据
            if coords_for_plot is not None:
                import plotly.graph_objects as go
                import numpy as np
                import matplotlib
                import matplotlib.pyplot as plt
                
                # 准备数据
                display_df = coords_for_plot.copy()
                display_df['主要细胞类型'] = predict_df.idxmax(axis=1).values
                display_df['主要比例'] = predict_df.max(axis=1).values
                
                # 计算绝对大小 (Pixel Size)
                # 基于实际数据范围归一化，确保差异可见
                p = display_df['主要比例'].values
                min_p, max_p = p.min(), p.max()
                
                # 归一化到 0-1
                normalized = (p - min_p) / (max_p - min_p + 1e-6)
                
                # 使用指数函数放大差异，映射到 8-25 像素
                # e^(2*x) 在 x=0 时为 1，x=1 时为 e^2≈7.39
                # 归一化后：(e^(2*x) - 1) / (e^2 - 1) 范围 0-1
                exp_normalized = (np.exp(2.0 * normalized) - 1) / (np.exp(2.0) - 1)
                pixel_sizes = 8 + exp_normalized * 17  # 范围 8-25
                
                display_df['pixel_size'] = pixel_sizes

                # 准备颜色
                unique_types = sorted(predict_df.columns.tolist())
                if len(unique_types) <= 10:
                    colors_list = plt.rcParams["axes.prop_cycle"].by_key()['color'][:len(unique_types)]
                else:
                    color_tab = plt.get_cmap('rainbow', len(unique_types))
                    colors_list = [matplotlib.colors.to_hex(x, keep_alpha=False) for x in color_tab(range(len(unique_types)))]
                color_map = dict(zip(unique_types, colors_list))

                # 创建 Figure
                fig = go.Figure()

                # 按类型分组添加 Traces
                # 这样每种类型都有独立的图例项，且颜色正确
                for cell_type in unique_types:
                    # 筛选该类型的数据
                    subset = display_df[display_df['主要细胞类型'] == cell_type]
                    
                    if len(subset) == 0:
                        continue
                        
                    # 构建悬停文本
                    # 注意：需要重新根据 subset 的 index 找到对应的详细比例
                    hover_texts = []
                    for idx in subset.index:
                        # 找到原始 predict_df 中的对应行
                        # 假设 coords_for_plot 的 index 和 predict_df 的 index 是一致的（在开头已经验证过）
                        row = predict_df.loc[idx]
                        sorted_row = row.sort_values(ascending=False)
                        text = f"<b>位置 {idx}</b><br>主要类型: {cell_type} ({subset.loc[idx, '主要比例']:.2%})<br>"
                        for ct, prop in sorted_row.head(hover_count).items():
                            text += f"{ct}: {prop:.2%}<br>"
                        hover_texts.append(text)

                    fig.add_trace(
                        go.Scatter(
                            x=subset['x'],
                            y=subset['y'],
                            mode='markers',
                            name=cell_type,
                            marker=dict(
                                color=color_map[cell_type],
                                size=subset['pixel_size'], # 这里传入的是绝对像素值
                                sizemode='diameter',       # 关键！直接解析为直径像素
                                opacity=0.9,
                                line=dict(width=0)         # 无描边
                            ),
                            hovertemplate='%{hovertext}<extra></extra>',
                            hovertext=hover_texts
                        )
                    )

                fig.update_layout(
                    height=650,
                    title='主要类型分布',
                    yaxis=dict(scaleanchor="x", scaleratio=1, visible=False, showgrid=False),
                    xaxis=dict(visible=False, showgrid=False),
                    legend=dict(
                        orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
                        itemclick="toggle", itemdoubleclick="toggleothers"
                    ),
                    dragmode='pan',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                
                # 处理全选/全不选按钮
                if deselect_all:
                    fig.for_each_trace(lambda trace: trace.update(visible='legendonly'))
                elif select_all:
                    fig.for_each_trace(lambda trace: trace.update(visible=True))
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 提示：点的大小直接反映置信度（指数级差异）。单击图例可隐藏/显示单个类型。")
            else:
                st.warning("无法显示交互式图表（坐标数据不匹配）")
        
        # --- Tab 3: 整体比例统计 ---
        with tabs[2]:
            st.subheader("📊 整体比例统计")
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
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 4: 单细胞类型热图 ---
        with tabs[3]:
            # 细胞类型选择器放在热图标签页内
            selected_type = st.selectbox(
                "🔬 选择要查看的细胞类型",
                cell_types,
                index=0
            )
            st.subheader(f"单细胞类型热图: {selected_type}")
            
            # 查找对应的热图
            heatmap_path = os.path.join(result_dir, f"{selected_type}.jpg")
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, use_container_width=True)
            else:
                if coords_for_plot is not None:
                     import plotly.express as px
                     plot_df = coords_for_plot.copy()
                     plot_df['proportion'] = predict_df[selected_type].values
                     fig = px.scatter(
                         plot_df, x='x', y='y', color='proportion',
                         color_continuous_scale='Viridis',
                         title=f'{selected_type} 空间分布',
                         size_max=15
                     )
                     fig.update_layout(height=600, yaxis=dict(scaleanchor="x", scaleratio=1))
                     st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无法显示热图")
        
        # --- Tab 5: 模型训练监控 ---
        with tabs[4]:
            st.subheader("模型训练监控")
            loss_path = os.path.join(result_dir, "Loss_function.jpg")
            if os.path.exists(loss_path):
                st.image(loss_path, use_container_width=True)
            else:
                st.warning("Loss 曲线文件不存在")
        
        # --- Tab 6: 详细数据表 ---
        with tabs[5]:
            st.subheader("详细数据表")
            st.dataframe(predict_df, use_container_width=True, height=400)
            
            # 下载按钮
            csv = predict_df.to_csv()
            st.download_button(
                label="📥 下载 CSV",
                data=csv,
                file_name="predict_result.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
