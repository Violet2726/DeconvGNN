"""
STdGCN 可视化系统入口
组织 Streamlit 界面布局与交互逻辑，包括侧边栏数据管理与主区域图表展示。
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import streamlit.components.v1 as components
import base64

# --- 兼容导入 (适配本地开发与 Streamlit Cloud 部署) ---
try:
    # 尝试作为模块导入 (当工作目录是项目根目录时)
    import visualization.styles as styles
    import visualization.data_loader as data_loader
    import visualization.utils as utils
except ImportError:
    # 尝试直接导入 (当工作目录是 visualization 目录时，例如 Streamlit Cloud 默认行为)
    import styles
    import data_loader
    import utils

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="DeconvGNN-Vis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# 注入自定义样式（强制按钮不换行、隐藏默认菜单等）
styles.inject_custom_css()



def main():
    """
    主函数：控制整体应用流程
    """
    
    # === 侧边栏区域：数据选择与管理 ===
    with st.sidebar:
        # 顶部标题
        st.markdown('<p class="main-header">DeconvGNN-Vis<br>空间转录组反卷积<br>可视化系统</p>', unsafe_allow_html=True)
        st.divider()
        
        # 调试工具：清除缓存
        if st.button("⚡ 重置系统", use_container_width=True, help="如果遇到数据加载问题，请点击此按钮重置"):
            st.cache_data.clear()
            st.rerun()
            
        st.divider()

        st.header("数据集")
        
        # 初始化会话状态 (Session State)
        if 'data_sources' not in st.session_state:
            # 初始为空，或者从配置读取预设
            st.session_state.data_sources = data_loader.DATA_DIRS.copy()
        
        if 'show_import' not in st.session_state:
            st.session_state.show_import = False
            
        # 1. 获取现有数据集列表
        options = list(st.session_state.data_sources.keys())
        
        # ------------------- 侧边栏逻辑：空状态处理 -------------------
        if not options:
            # 如果没有数据，且没在导入，显式提示
            selected_dataset_name = None
            result_dir = None
        else:
            # ------------------- 侧边栏逻辑：数据集选择器 -------------------
            # 下拉菜单 (单独一行，保证宽度和美观)
            selected_dataset_name = st.selectbox(
                "选择数据集",
                options=options,
                index=0,
                label_visibility="visible",
                key="dataset_selector"  # 绑定 state 以便编程控制选中项
            )
            result_dir = st.session_state.data_sources[selected_dataset_name]


        # ------------------- 侧边栏逻辑：功能按钮 -------------------
        # 两列布局：删除 | 导入
        col_del, col_add = st.columns(2)
        
        with col_del:
            # 仅当有选中数据时才启用删除
            if selected_dataset_name:
                if st.button("🗑️ 移除", use_container_width=True, help="删除当前选中的数据集"):
                    # 如果删除的是上传的数据，同时清理 uploaded_data
                    if st.session_state.data_sources.get(selected_dataset_name) == "__UPLOADED__":
                        if 'uploaded_data' in st.session_state:
                            del st.session_state.uploaded_data
                    del st.session_state.data_sources[selected_dataset_name]
                    # 删除当前选中项后，清除 selector 状态防止报错
                    if "dataset_selector" in st.session_state:
                        del st.session_state.dataset_selector
                    st.rerun()
            else:
                 st.button("🗑️ 删除", disabled=True, use_container_width=True)

        with col_add:
            # 导入/取消导入 切换按钮
            btn_label = "✖️ 取消" if st.session_state.show_import and options else "✨ 导入"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_import = not st.session_state.show_import
                st.rerun()



        st.divider()

        # ------------------- 侧边栏逻辑：导入面板 -------------------
        # 嵌入式显示，点击导入后展开
        if st.session_state.show_import:
            with st.container():
                st.markdown("#### <i class='fa-solid fa-cloud-arrow-up'></i> 导入新数据", unsafe_allow_html=True)
                
                # 检测运行环境
                is_cloud = utils.is_cloud_environment()
                
                if is_cloud:
                    # ===== 云端模式：使用文件上传 =====
                    
                    uploaded_files = st.file_uploader(
                        "上传数据文件",
                        type=["csv"],
                        accept_multiple_files=True,
                        help="请上传 predict_result.csv 和 coordinates.csv",
                        key="cloud_uploader"
                    )
                    
                    if uploaded_files and len(uploaded_files) >= 1:
                        # 检查是否包含必需文件
                        file_names = [f.name.lower() for f in uploaded_files]
                        has_predict = any("predict" in name for name in file_names)
                        
                        if has_predict:
                            new_name = st.text_input("数据集命名", value="上传的数据集")
                            
                            def on_upload_confirm():
                                if new_name:
                                    # 立即解析并缓存数据，实现持久化（防止 rerun 后文件流丢失）
                                    pdf, cdf = data_loader.load_from_uploaded_files(uploaded_files)
                                    if pdf is not None:
                                        st.session_state.uploaded_data_cache = {
                                            'predict_df': pdf,
                                            'coords': cdf
                                        }
                                        st.session_state.data_sources[new_name] = "__UPLOADED__"
                                        st.session_state.dataset_selector = new_name
                                        st.session_state.show_import = False
                                    else:
                                        st.toast("❌ 数据解析失败，请检查文件格式", icon="❌")
                                else:
                                    st.error("请输入名称")
                            
                            st.button("✅ 确认添加", type="primary", use_container_width=True, on_click=on_upload_confirm)
                        else:
                            st.warning("⚠️ 请确保上传的文件包含 `predict_result.csv`")
                    
                    with st.expander("📋 文件要求", expanded=False):
                        st.markdown("""
                        **必需文件：**
                        - `predict_result.csv` - 反卷积预测结果
                        - `coordinates.csv` - 空间坐标数据
                        """)
                else:
                    # ===== 本地模式：使用文件夹选择 =====
                    if 'temp_import_path' not in st.session_state:
                        st.session_state.temp_import_path = ""
                        
                    col_path, col_browse = st.columns([3, 1])
                    with col_path:
                         st.text_input("路径", value=st.session_state.temp_import_path, disabled=True, label_visibility="collapsed", placeholder="请选择文件夹...")
                    with col_browse:
                        if st.button("📂", key="btn_browse_folder", use_container_width=True):
                            folder = utils.open_folder_dialog()
                            if folder:
                                st.session_state.temp_import_path = folder
                                st.rerun()
                    
                    # 确认逻辑
                    if st.session_state.temp_import_path:
                        raw_path = st.session_state.temp_import_path
                        
                        # 智能路径推断：检查根目录和 results 子目录
                        valid_path = None
                        if os.path.exists(os.path.join(raw_path, "predict_result.csv")):
                            valid_path = raw_path
                        elif os.path.exists(os.path.join(raw_path, "results", "predict_result.csv")):
                            valid_path = os.path.join(raw_path, "results")
                            
                        if valid_path:
                            default_name = os.path.basename(raw_path)
                            new_name = st.text_input("数据集命名", value=default_name)
                            
                            def on_add_confirm():
                                if new_name:
                                    st.session_state.data_sources[new_name] = valid_path
                                    st.session_state.dataset_selector = new_name
                                    st.session_state.show_import = False
                                    st.session_state.temp_import_path = ""
                                else:
                                    st.error("请输入名称")

                            st.button("✅ 确认添加", type="primary", use_container_width=True, on_click=on_add_confirm)
                            
                            with st.expander("📋 文件要求", expanded=False):
                                st.markdown("""
                                **必需文件：**
                                - `predict_result.csv` - 反卷积预测结果
                                - `coordinates.csv` - 空间坐标数据
                                """)
                        else:
                            st.error(f"❌ 未找到关键文件 `predict_result.csv`。\n请确保选择的目录（或其 `results` 子目录）包含该文件。")
                
                st.divider()
        
    # === 主内容区域 ===
    
    # 1. 全局数据检查
    if result_dir is None:
        # 1. 引导箭头 (仅在未导入数据时显示)
        st.markdown('<div class="sidebar-hint"><i class="fa-solid fa-angles-left" style="font-size:3rem; color:#00f260; filter: drop-shadow(0 0 10px #00f260);"></i></div>', unsafe_allow_html=True)
        
        # 2. 炫技首页内容 (使用无缩进字符串，防止被识别为代码块)
        # 2. 炫技首页内容 (使用无缩进字符串，防止被识别为代码块)
        banner_base64 = utils.get_base64_image(str(utils.BANNER_PATH))
        banner_src = f"data:image/png;base64,{banner_base64}" if banner_base64 else "https://images.unsplash.com/photo-1628595351029-c2bf17511435?q=80&w=2000&auto=format&fit=crop"

        landing_html = styles.get_landing_page_html(banner_src)
        st.markdown(landing_html, unsafe_allow_html=True)
        return
        
    # 2. 加载数据
    if result_dir == "__UPLOADED__":
        # 云端模式：从 Session State 缓存加载
        if 'uploaded_data_cache' in st.session_state:
            predict_df = st.session_state.uploaded_data_cache['predict_df']
            coords = st.session_state.uploaded_data_cache['coords']
        else:
            st.error("❌ 上传的数据缓存已失效，请重新上传")
            return
            return
    else:
        # 本地模式：从文件路径加载
        predict_df, coords = data_loader.load_results(result_dir)
    
    if predict_df is not None:
        cell_types = data_loader.get_cell_types(predict_df)
    else:
        st.error("❌ 未找到结果文件")
        st.info(f"请先运行 Tutorial.py 生成结果，或重新上传数据文件")
        return
    
    # 3. 顶部统计仪表盘
    if predict_df is not None:
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
        
        # ========== 图表缓存系统 (基于 session_state) ==========
        # 使用数据集名称作为缓存键，切换回已加载过的数据集时瞬间显示
        if 'figure_cache' not in st.session_state:
            st.session_state.figure_cache = {}
        
        # 当前数据集的缓存键前缀
        cache_prefix = f"{selected_dataset_name}_"
        
        # 创建 Tab 标签页 (使用更现代的 Emoji)
        tabs = st.tabs([
            "🧩 空间组分图谱", 
            "🔍 优势亚群分布", 
            "📊 细胞比例概览", 
            "🌡️ 单细胞热度图", 
            "📑 原始数据详单"
        ])
        
        # --- Tab 1: 空间组成分布 (Plotly Scatter + 饼图背景) ---
        with tabs[0]:
            # st.subheader 已移除，使用图表内部标题
            # 数据准备
            coords_for_plot = coords

            # 设置栏
            with st.expander("⚙️ 视图配置", expanded=False):
                hover_count_tab1 = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)), key="tab1_hover")

            if coords_for_plot is not None:
                # 1. 加载或生成背景图 (优先使用缓存)
                bg_cache_key = f"{cache_prefix}bg_img"
                
                # 优先级: session_state 缓存 > 磁盘文件 > 现场生成
                if bg_cache_key in st.session_state.figure_cache:
                    # 从 session_state 缓存读取
                    cached_bg = st.session_state.figure_cache[bg_cache_key]
                    bg_img = cached_bg['img']
                    xlim = cached_bg['xlim']
                    ylim = cached_bg['ylim']
                else:
                    bg_img = None
                    xlim, ylim = None, None
                    
                    # 尝试从磁盘读取 (仅对本地数据集有效)
                    precomputed_img_path = os.path.join(result_dir, "interactive_pie_background.png")
                    precomputed_meta_path = os.path.join(result_dir, "interactive_pie_bounds.json")
                    
                    if result_dir != "__UPLOADED__" and os.path.exists(precomputed_img_path) and os.path.exists(precomputed_meta_path):
                        from PIL import Image
                        import json
                        bg_img = Image.open(precomputed_img_path)
                        with open(precomputed_meta_path, 'r') as f:
                            metadata = json.load(f)
                            xlim = metadata['xlim']
                            ylim = metadata['ylim']
                    else:
                        # 现场生成
                        progress_bar = st.progress(0, text="⏳ 首次加载，正在生成饼图背景...")
                        status_text = st.empty()
                        
                        def update_progress(pct, msg):
                            progress_bar.progress(pct, text=f"⏳ {msg}")
                        
                        bg_img, (xlim, ylim) = utils.generate_clean_pie_chart(
                            predict_df, coords_for_plot, None, 
                            progress_callback=update_progress
                        )
                        
                        # 保存到磁盘 (仅本地数据集)
                        if result_dir != "__UPLOADED__":
                            utils.save_pie_chart_background(bg_img, xlim, ylim, result_dir)
                        
                        progress_bar.empty()
                        status_text.empty()
                    
                    # 存入 session_state 缓存
                    st.session_state.figure_cache[bg_cache_key] = {
                        'img': bg_img,
                        'xlim': xlim,
                        'ylim': ylim
                    }
                
                # 2. 生成交互式图表 (使用缓存)
                tab1_cache_key = f"{cache_prefix}tab1_{hover_count_tab1}"
                
                if tab1_cache_key not in st.session_state.figure_cache:
                    cell_type_color_map = utils.get_color_map(predict_df.columns.tolist(), predict_df)
                    fig = utils.generate_plotly_scatter(
                        coords_for_plot, predict_df, hover_count_tab1, 
                        bg_img, (xlim, ylim), cell_type_color_map
                    )
                    st.session_state.figure_cache[tab1_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab1_cache_key]
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True, 'staticPlot': False})
                st.caption("💡 说明：此图背景为多色饼图，展示每个位置的细胞组成；鼠标悬停可查看具体比例数据。")
            else:
                 st.warning("缺少坐标数据，无法生成交互式图表。显示静态预览：")
                 pie_plot_path = os.path.join(result_dir, "predict_results_pie_plot.jpg")
                 if os.path.exists(pie_plot_path):
                     st.image(pie_plot_path, use_container_width=True)

        # --- Tab 2: 主要类型分布 (Dominant Scatter) ---
        with tabs[1]:
            # st.subheader 已移除
            
            with st.expander("⚙️ 视图配置", expanded=False):
                hover_count = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)), key="tab2_hover")
                
            if coords_for_plot is not None:
                # 使用缓存系统
                tab2_cache_key = f"{cache_prefix}tab2_{hover_count}"
                
                if tab2_cache_key not in st.session_state.figure_cache:
                    plot_predict_df = predict_df
                    plot_coords = coords_for_plot
                    unique_types = sorted(predict_df.columns.tolist())
                    color_map = utils.get_color_map(unique_types, predict_df)
                    
                    fig = utils.generate_dominant_scatter(
                        plot_coords, plot_predict_df, hover_count, color_map
                    )
                    st.session_state.figure_cache[tab2_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab2_cache_key]
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                st.caption(
                    """
                    🖱️ 图例操作说明：
                    -  单击：选中或取消选中该类型
                    -  双击（高亮时）：只显示该类型（独显模式）
                    -  双击（灰色时）：全选所有类型（恢复显示）
                    """)
            else:
                st.warning("无法显示交互式图表（坐标数据不匹配）")
        
        # --- Tab 3: 整体比例统计 (Bar Chart) ---
        with tabs[2]:
            tab3_cache_key = f"{cache_prefix}tab3"
            
            if tab3_cache_key not in st.session_state.figure_cache:
                fig = utils.generate_proportion_bar(predict_df)
                st.session_state.figure_cache[tab3_cache_key] = fig
            else:
                fig = st.session_state.figure_cache[tab3_cache_key]
            
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 4: 单细胞类型热图 (Heatmap) ---
        with tabs[3]:
            selected_type = st.selectbox("🔬 选择要查看的细胞类型", cell_types, index=0)

            if coords_for_plot is not None:
                tab4_cache_key = f"{cache_prefix}tab4_{selected_type}"
                
                if tab4_cache_key not in st.session_state.figure_cache:
                    fig = utils.generate_heatmap(coords_for_plot, predict_df, selected_type)
                    st.session_state.figure_cache[tab4_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab4_cache_key]
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
            else:
                # 尝试显示静态图 fallback
                heatmap_path = os.path.join(result_dir, f"{selected_type}.jpg")
                if os.path.exists(heatmap_path):
                    st.image(heatmap_path, use_container_width=True)
                else:
                    st.warning("暂无该类型的坐标数据或静态图片。")
        
        # --- Tab 5: 详细数据表 (Table) ---
        with tabs[4]:
            st.subheader("📑 原始数据详单")
            st.dataframe(predict_df, use_container_width=True, height=400)

if __name__ == "__main__":
    main()
