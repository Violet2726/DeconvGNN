"""
DeconvGNN-Vis 可视化系统入口
该模块负责构建基于 Streamlit 的 Web 界面，包括数据集管理、实时图表渲染及交互逻辑。
"""

import streamlit as st
import pandas as pd
import os


# --- 跨环境导入适配 (支持本地开发与 Streamlit Cloud) ---
try:
    import visualization.styles as styles
    import visualization.data_loader as data_loader
    import visualization.utils as utils
except ImportError:
    import styles
    import data_loader
    import utils

# --- 页面全局配置 ---
st.set_page_config(
    page_title="DeconvGNN-Vis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed" # 初始收起侧边栏以展示欢迎页
)



# 注入自定义样式（强制按钮不换行、隐藏默认菜单等）
styles.inject_custom_css()



def main():
    """
    应用核心入口函数，控制整体业务逻辑与界面流转。
    """
    
    # === 侧边栏区域：数据源管理 ===
    with st.sidebar:
        st.markdown('<p class="main-header">DeconvGNN-Vis<br>空间转录组反卷积<br>可视化系统</p>', unsafe_allow_html=True)
        st.divider()
        
        # 系统重置工具
        if st.button("⚡ 重置系统", use_container_width=True, help="清除所有缓存并重新加载应用"):
            st.cache_data.clear()
            st.rerun()
            
        st.divider()
        st.header("数据集管理")
        
        # 初始化会话数据源
        if 'data_sources' not in st.session_state:
            st.session_state.data_sources = data_loader.DATA_DIRS.copy()
        
        if 'show_import' not in st.session_state:
            st.session_state.show_import = False
            
        # 数据集列表获取与选择逻辑
        options = list(st.session_state.data_sources.keys())
        
        # ------------------- 侧边栏逻辑：空状态处理 -------------------
        if not options:
            # 如果没有数据，且没在导入，显式提示
            selected_dataset_name = None
            result_dir = None
        else:
            # ------------------- 侧边栏逻辑：数据集选择器 -------------------
            # 数据集下拉选择器
            selected_dataset_name = st.selectbox(
                "选择当前数据集",
                options=options,
                index=0,
                label_visibility="visible",
                key="dataset_selector"
            )
            result_dir = st.session_state.data_sources[selected_dataset_name]


        # 数据集操作工具栏 (删除与新增)
        col_del, col_add = st.columns(2)
        
        with col_del:
            if selected_dataset_name:
                if st.button("🗑️ 移除", use_container_width=True, help="从当前会话中移除该数据集"):
                    if st.session_state.data_sources.get(selected_dataset_name) == "__UPLOADED__":
                        if 'uploaded_data' in st.session_state:
                            del st.session_state.uploaded_data
                    del st.session_state.data_sources[selected_dataset_name]
                    # 重置选择器状态
                    if "dataset_selector" in st.session_state:
                        del st.session_state.dataset_selector
                    st.rerun()
            else:
                 st.button("🗑️ 移除", disabled=True, use_container_width=True)
 
        with col_add:
            # 切换导入面板显示状态
            btn_label = "✖️ 取消" if st.session_state.show_import and options else "✨ 导入"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_import = not st.session_state.show_import
                st.rerun()



        st.divider()

        # 数据导入交互面板
        if st.session_state.show_import:
            with st.container():
                st.markdown("#### <i class='fa-solid fa-cloud-arrow-up'></i> 导入新项目", unsafe_allow_html=True)
                is_cloud = utils.is_cloud_environment()
                
                if is_cloud:
                    # 云端部署模式：基于文件上传的数据加载
                    
                    uploaded_files = st.file_uploader(
                        "上传数据文件",
                        type=["csv"],
                        accept_multiple_files=True,
                        help="请上传 predict_result.csv 和 coordinates.csv",
                        key="cloud_uploader"
                    )
                    
                    if uploaded_files:
                        file_names = [f.name.lower() for f in uploaded_files]
                        if any("predict" in name for name in file_names):
                            new_name = st.text_input("数据集显示名称", value="新上传数据集")
                            
                            def on_upload_confirm():
                                if new_name:
                                    # 解析数据并持久化到 Session Cache
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
                                        st.toast("❌ 数据解析失败，请检查 CSV 格式", icon="❌")
                                else:
                                    st.error("请输入名称")
                            
                            st.button("✅ 确认上传", type="primary", use_container_width=True, on_click=on_upload_confirm)
                        else:
                            st.warning("⚠️ 必需文件缺失：请务必上传 `predict_result.csv`")
                    
                    with st.expander("📋 文件规范", expanded=False):
                        st.markdown("""
                        **必须上传以下文件：**
                        - `predict_result.csv`: 模型预测结果（细胞占比）
                        - `coordinates.csv`: 空间位点坐标
                        """)
                else:
                    # 本地开发模式：基于文件路径的智能导入
                    if 'temp_import_path' not in st.session_state:
                         st.session_state.temp_import_path = ""
                        
                    col_path, col_browse = st.columns([3, 1])
                    with col_path:
                         st.text_input("本地路径", value=st.session_state.temp_import_path, disabled=True, label_visibility="collapsed")
                    with col_browse:
                        if st.button("📂", use_container_width=True):
                            folder = utils.open_folder_dialog()
                            if folder:
                                st.session_state.temp_import_path = folder
                                st.rerun()
                    
                    # 确认逻辑
                    if st.session_state.temp_import_path:
                        raw_path = st.session_state.temp_import_path
                        
                        # 检测路径有效性（支持根目录或 results 子目录）
                        valid_path = None
                        if os.path.exists(os.path.join(raw_path, "predict_result.csv")):
                            valid_path = raw_path
                        elif os.path.exists(os.path.join(raw_path, "results", "predict_result.csv")):
                            valid_path = os.path.join(raw_path, "results")
                            
                        if valid_path:
                            default_name = os.path.basename(raw_path)
                            new_name = st.text_input("数据集显示名称", value=default_name)
                            
                            def on_add_confirm():
                                if new_name:
                                    st.session_state.data_sources[new_name] = valid_path
                                    st.session_state.dataset_selector = new_name
                                    st.session_state.show_import = False
                                    st.session_state.temp_import_path = ""
                                else:
                                    st.error("请输入名称")
 
                            st.button("✅ 确认导入", type="primary", use_container_width=True, on_click=on_add_confirm)
                        else:
                            st.error(f"❌ 目录无效：未能在该路径下找到 `predict_result.csv`。")
                st.divider()
 
    # === 主界面展示区 ===
    
    # 无数据场景：展示欢迎页与系统简介
    if result_dir is None:
        # 指向侧边栏的交互指引
        st.markdown('<div class="sidebar-hint"><i class="fa-solid fa-angles-left" style="font-size:3rem; color:#00f260; filter: drop-shadow(0 0 10px #00f260);"></i></div>', unsafe_allow_html=True)
        
        # 首页视觉渲染 (基于 Assets 图片与动态样式)
        banner_base64 = utils.get_base64_image(str(utils.BANNER_PATH))
        banner_src = f"data:image/png;base64,{banner_base64}" if banner_base64 else ""
 
        st.markdown(styles.get_landing_page_html(banner_src), unsafe_allow_html=True)
        return
        
    # 有效数据场景：执行数据流加载
    if result_dir == "__UPLOADED__":
        # 云端部署加载逻辑：通过 Session State 恢复
        if 'uploaded_data_cache' in st.session_state:
            predict_df = st.session_state.uploaded_data_cache['predict_df']
            coords = st.session_state.uploaded_data_cache['coords']
        else:
            st.error("❌ 会话过期：上传的数据已失效，请重新上传文件。")
            return
    else:
        # 本地开发加载逻辑：通过文件系统读取
        predict_df, coords = data_loader.load_results(result_dir)
    
    if predict_df is not None:
        cell_types = data_loader.get_cell_types(predict_df)
    else:
        st.error("❌ 加载失败：未能解析反卷积结果文件。")
        st.info("请确保输出目录完整，或尝试重新导入数据。")
        return
    
    # 核心指标看板渲染
    if predict_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("空间观测位点", f"{len(predict_df):,}")
        with col2:
            st.metric("检测细胞类型", len(cell_types))
        with col3:
            st.metric("丰度最高类型", predict_df.mean().idxmax())
        with col4:
            st.metric("平均占比峰值", f"{predict_df[predict_df.mean().idxmax()].mean():.2%}")
        
        st.divider()
        
        # ========== 模块化图表视图渲染 ==========
        
        # 初始化图表缓存系统 (基于 Session State 确保切换 Tab 无需重算)
        if 'figure_cache' not in st.session_state:
            st.session_state.figure_cache = {}
        
        # 当前数据集的缓存键前缀
        cache_prefix = f"{selected_dataset_name}_"
        
        # 构建可视化菜单
        tabs = st.tabs([
            "🧩 空间组分图谱", 
            "🔍 优势亚群分布", 
            "📊 细胞比例概览", 
            "🌡️ 单细胞热度图", 
            "📑 原始数据详单"
        ])
        
        # --- 视图 1: 空间组成分布 (360° 交互式散点饼图) ---
        with tabs[0]:
            # st.subheader 已移除，使用图表内部标题
            # 数据准备
            coords_for_plot = coords

            # 动态视图参数配置
            with st.expander("⚙️ 映射策略配置", expanded=False):
                hover_count_tab1 = st.slider("悬停详情数量", 3, len(cell_types), min(6, len(cell_types)), key="tab1_hover")

            if coords_for_plot is not None:
                # 背景层加载逻辑 (智能缓存: Session -> Disk -> Memory Generate)
                bg_cache_key = f"{cache_prefix}bg_img"
                
                # 优先级: session_state 缓存 > 磁盘文件 > 现场生成
                if bg_cache_key in st.session_state.figure_cache:
                    cached_bg = st.session_state.figure_cache[bg_cache_key]
                    bg_img = cached_bg['img']
                    xlim, ylim = cached_bg['xlim'], cached_bg['ylim']
                else:
                    # 首次访问执行密集计算流水线
                    progress_bar = st.progress(0, text="🧪 正在通过并行管道计算空间饼图轨迹...")
                    
                    def update_progress(pct, msg):
                        progress_bar.progress(pct, text=f"⏳ {msg}")
                    
                    bg_img, (xlim, ylim) = utils.get_or_generate_pie_background(
                        predict_df, coords_for_plot, result_dir, 
                        progress_callback=update_progress
                    )
                    progress_bar.empty()
                    
                    # 更新持久化缓存
                    st.session_state.figure_cache[bg_cache_key] = {'img': bg_img, 'xlim': xlim, 'ylim': ylim}
                
                # 前景交互层渲染
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
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                st.caption("💡 视图说明：背景层展示各观测位点的多组分构成；悬停可探索亚细胞级占比详情。")
            else:
                 st.warning("⚠️ 坐标数据缺失或不兼容：无法生成空间拓扑图。")
                 
        # --- 视图 2: 优势亚群分布 (WebGl 加速散点图) ---
        with tabs[1]:
            with st.expander("⚙️ 渲染参数配置", expanded=False):
                hover_count = st.slider("悬停详情数量", 3, len(cell_types), min(6, len(cell_types)), key="tab2_hover")
                
            if coords_for_plot is not None:
                tab2_cache_key = f"{cache_prefix}tab2_{hover_count}"
                
                if tab2_cache_key not in st.session_state.figure_cache:
                    color_map = utils.get_color_map(predict_df.columns.tolist(), predict_df)
                    fig = utils.generate_dominant_scatter(coords_for_plot, predict_df, hover_count, color_map)
                    st.session_state.figure_cache[tab2_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab2_cache_key]
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("🖱️ 交互贴士：通过点击右侧图例可进行细胞类型筛选；双击可切换独显/全选模式。")
            else:
                st.warning("⚠️ 数据异常：该数据集无法进行优势亚群聚类映射。")
        
        # --- 视图 3: 全局比例统计 (汇总柱状图) ---
        with tabs[2]:
            tab3_cache_key = f"{cache_prefix}tab3"
            
            if tab3_cache_key not in st.session_state.figure_cache:
                fig = utils.generate_proportion_bar(predict_df)
                st.session_state.figure_cache[tab3_cache_key] = fig
            else:
                fig = st.session_state.figure_cache[tab3_cache_key]
            
            st.plotly_chart(fig, use_container_width=True)

        # --- 视图 4: 空间表达热力图 (基于选定类型) ---
        with tabs[3]:
            selected_type = st.selectbox("🔬 检索目标细胞亚群", cell_types, index=0)

            if coords_for_plot is not None:
                tab4_cache_key = f"{cache_prefix}tab4_{selected_type}"
                
                if tab4_cache_key not in st.session_state.figure_cache:
                    fig = utils.generate_heatmap(coords_for_plot, predict_df, selected_type)
                    st.session_state.figure_cache[tab4_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab4_cache_key]
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 提示：缺少该样本的空间坐标。")
        
        # --- 视图 5: 数据详单分析 (交互式表格) ---
        with tabs[4]:
            st.markdown("#### 📑 反卷积预测原始指标矩阵")
            st.dataframe(predict_df, use_container_width=True, height=500)

if __name__ == "__main__":
    main()
