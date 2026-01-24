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
            
            # 显示设置
            st.header("⚙️ 显示设置")
            hover_count = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)))
            
            st.divider()
            
            # 图例控制按钮
            st.header("👁️ 图例控制")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                select_all = st.button("全选", use_container_width=True)
            with col_btn2:
                deselect_all = st.button("全不选", use_container_width=True)
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
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 空间分布饼图", "🔥 细胞类型热图", "📈 训练曲线", "📋 数据表格"])
        
        with tab1:
            st.subheader("空间分布图")
            
            # 显示模式切换
            display_mode = st.radio(
                "选择显示模式",
                ["🎨 饼图模式（多色比例）", "🔍 交互模式（悬停查看）"],
                horizontal=True,
                index=1
            )
            
            if display_mode == "🎨 饼图模式（多色比例）":
                # 显示静态饼图
                pie_plot_path = os.path.join(result_dir, "predict_results_pie_plot.jpg")
                if os.path.exists(pie_plot_path):
                    st.image(pie_plot_path, use_container_width=True)
                    st.caption("💡 此图为静态图片，每个点的颜色比例代表不同细胞类型组成")
                else:
                    st.warning("饼图文件不存在，请先运行 Tutorial.py 生成结果")
            else:
                # 交互模式
                # 尝试加载坐标数据
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
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # 构建带有悬停信息的数据
                    plot_df = coords_for_plot.copy()
                    
                    # 找出每个点的主要细胞类型（用于着色）
                    plot_df['主要细胞类型'] = predict_df.idxmax(axis=1).values
                    plot_df['主要比例'] = predict_df.max(axis=1).values
                    
                    # 构建悬停文本：显示前 N 种细胞类型的比例
                    hover_texts = []
                    for idx in range(len(predict_df)):
                        row = predict_df.iloc[idx]
                        # 按比例排序，显示前 hover_count 个
                        sorted_row = row.sort_values(ascending=False)
                        text = f"<b>位置 {predict_df.index[idx]}</b><br>"
                        for cell_type, proportion in sorted_row.head(hover_count).items():
                            bar = "█" * int(proportion * 20)  # 简单的条形图
                            text += f"{cell_type}: {proportion:.2%}<br>"
                        hover_texts.append(text)
                    
                    plot_df['hover_text'] = hover_texts
                    
                    # 创建交互式散点图
                    fig = px.scatter(
                        plot_df,
                        x='x', y='y',
                        color='主要细胞类型',
                        size='主要比例',
                        size_max=15,
                        hover_name='hover_text',
                        title='空间分布图（鼠标悬停查看详细比例）'
                    )
                    
                    fig.update_traces(
                        hovertemplate='%{hovertext}<extra></extra>',
                        hovertext=hover_texts
                    )
                    
                    fig.update_layout(
                        height=650,
                        yaxis=dict(scaleanchor="x", scaleratio=1),
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02,
                            itemclick="toggle",
                            itemdoubleclick="toggleothers"
                        )
                    )
                    
                    # 处理全选/全不选按钮
                    if deselect_all:
                        for trace in fig.data:
                            trace.visible = "legendonly"
                    elif select_all:
                        for trace in fig.data:
                            trace.visible = True
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("💡 提示：单击图例可隐藏/显示单个细胞类型，双击可只显示该类型")
                else:
                    st.warning("无法显示交互式图表（坐标数据不匹配）")
        
        with tab2:
            # 细胞类型选择器放在热图标签页内
            selected_type = st.selectbox(
                "🔬 选择要查看的细胞类型",
                cell_types,
                index=0
            )
            st.subheader(f"细胞类型热图: {selected_type}")
            
            # 查找对应的热图
            heatmap_path = os.path.join(result_dir, f"{selected_type}.jpg")
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, use_container_width=True)
            else:
                # 显示交互式散点图
                if coords is not None:
                    import plotly.express as px
                    
                    plot_df = coords.copy()
                    plot_df['proportion'] = predict_df[selected_type].values
                    
                    fig = px.scatter(
                        plot_df,
                        x='x', y='y',
                        color='proportion',
                        color_continuous_scale='Viridis',
                        title=f'{selected_type} 空间分布',
                        size_max=point_size
                    )
                    fig.update_layout(
                        height=600,
                        yaxis=dict(scaleanchor="x", scaleratio=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("无法显示热图（坐标数据不匹配）")
        
        with tab3:
            st.subheader("模型训练曲线")
            loss_path = os.path.join(result_dir, "Loss_function.jpg")
            if os.path.exists(loss_path):
                st.image(loss_path, use_container_width=True)
            else:
                st.warning("Loss 曲线文件不存在")
        
        with tab4:
            st.subheader("预测结果数据表")
            st.dataframe(predict_df, use_container_width=True, height=400)
            
            # 下载按钮
            csv = predict_df.to_csv()
            st.download_button(
                label="📥 下载 CSV",
                data=csv,
                file_name="predict_result.csv",
                mime="text/csv"
            )
        
        # 第三行：细胞类型比例统计
        st.divider()
        st.subheader("📊 细胞类型平均比例")
        
        import plotly.express as px
        mean_proportions = predict_df.mean().sort_values(ascending=True)
        fig = px.bar(
            x=mean_proportions.values,
            y=mean_proportions.index,
            orientation='h',
            labels={'x': '平均比例', 'y': '细胞类型'},
            color=mean_proportions.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
