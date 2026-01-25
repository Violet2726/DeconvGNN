"""
STdGCN 空间转录组反卷积可视化系统
主应用入口文件

此文件负责组织主要的 Streamlit 界面布局和交互逻辑。
包含：
1. 页面配置与初始化
2. 全局样式注入 (from styles.py)
3. 侧边栏：数据集选择与管理 (from data_loader.py)
4. 主内容区：数据展示与可视化选项卡
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# --- 本地模块导入 ---
# styles: 负责所有 CSS 样式定义和注入
import visualization_app.styles as styles
# data_loader: 负责数据目录管理、文件读取和缓存
import visualization_app.data_loader as data_loader

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="STdGCN 可视化系统",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
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
        st.markdown('<p class="main-header">🧬 STdGCN<br>空间转录组反卷积<br>可视化系统</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">基于图神经网络的<br>细胞类型反卷积结果展示</p>', unsafe_allow_html=True)
        st.divider()

        st.header("📊 数据选择")
        
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
                "当前数据集",
                options,
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
                if st.button("🗑️ 删除", use_container_width=True, help="删除当前选中的数据集"):
                    del st.session_state.data_sources[selected_dataset_name]
                    # 删除当前选中项后，清除 selector 状态防止报错
                    if "dataset_selector" in st.session_state:
                        del st.session_state.dataset_selector
                    st.rerun()
            else:
                 st.button("🗑️ 删除", disabled=True, use_container_width=True)

        with col_add:
            # 导入/取消导入 切换按钮
            btn_label = "❌ 取消" if st.session_state.show_import and options else "📂 导入"
            if st.button(btn_label, use_container_width=True):
                st.session_state.show_import = not st.session_state.show_import
                st.rerun()

        st.divider()

        # ------------------- 侧边栏逻辑：导入面板 -------------------
        # 嵌入式显示，点击导入后展开
        if st.session_state.show_import:
            with st.container():
                st.markdown("#### 📥 导入新数据")
                
                if 'temp_import_path' not in st.session_state:
                    st.session_state.temp_import_path = ""
                    
                def open_folder_dialog():
                    """调用 tkinter 打开系统文件夹选择框"""
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                        root = tk.Tk()
                        root.withdraw()
                        root.attributes('-topmost', True)
                        folder = filedialog.askdirectory(title="选择 STdGCN 输出目录")
                        root.destroy()
                        return folder
                    except:
                        return None

                col_path, col_browse = st.columns([3, 1])
                with col_path:
                     st.text_input("路径", value=st.session_state.temp_import_path, disabled=True, label_visibility="collapsed", placeholder="请选择文件夹...")
                with col_browse:
                    if st.button("浏览", key="btn_browse_folder", use_container_width=True):
                        folder = open_folder_dialog()
                        if folder:
                            st.session_state.temp_import_path = folder
                            st.rerun()
                
                # 确认逻辑
                if st.session_state.temp_import_path:
                    import_path = st.session_state.temp_import_path
                    if os.path.exists(os.path.join(import_path, "predict_result.csv")):
                        default_name = os.path.basename(import_path)
                        new_name = st.text_input("数据集命名", value=default_name)
                        
                        # 定义回调函数，在按钮点击时直接修改 state
                        def on_add_confirm():
                            if new_name:
                                st.session_state.data_sources[new_name] = import_path
                                # 自动选中新添加的数据集
                                st.session_state.dataset_selector = new_name
                                st.session_state.show_import = False
                                st.session_state.temp_import_path = ""
                            else:
                                st.error("请输入名称")

                        st.button("➕ 确认添加", type="primary", use_container_width=True, on_click=on_add_confirm)
                        
                        with st.expander("查看数据要求", expanded=False):
                            st.markdown("""
                            **必需文件**：`predict_result.csv`  
                            **可选文件**：`coordinates.csv`
                            """)
                    else:
                        st.error("❌ 缺少 predict_result.csv")
                
                st.divider()
        
    # === 主内容区域 ===
    
    # 1. 全局数据检查
    if result_dir is None:
        st.title("欢迎使用 STdGCN 可视化系统")
        st.info("👈 请在左侧 **侧边栏** 导入数据以开始使用")
        return
        
    # 2. 加载数据 (使用 data_loader 模块，带缓存)
    predict_df, coords = data_loader.load_results(result_dir)
    
    if predict_df is not None:
        cell_types = data_loader.get_cell_types(predict_df)
    else:
        st.error("❌ 未找到结果文件")
        st.info(f"请先运行 Tutorial.py 生成结果")
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
        
        # 4. 可视化选项卡
        tabs = st.tabs([
            "🎨 空间组成分布", 
            "🔍 主要类型分布", 
            "📊 整体比例统计", 
            "🔥 单细胞类型热图", 
            "📈 详细数据表"
        ])
        
        # --- Tab 1: 空间组成分布 (Plotly Scatter + 饼图背景) ---
        with tabs[0]:
            st.subheader("空间组成分布 (多色饼图)")
            
            # 引入依赖 (局部引入以优化启动速度)
            import visualization_app.utils as utils
            
            # 检查坐标数据 (逻辑需要在 data_loader 中处理吗？暂时保持在这里因为涉及 specific logic)
            # 为了更好的逻辑分离，理想情况下应该把这部分也移出去，但现在主要任务是重构app.py结构
            
            # 使用 data_loader 提供的 coords 即可，它已经处理好了查找逻辑
            coords_for_plot = coords

            # 添加设置栏
            with st.expander("🛠️ 设置", expanded=False):
                hover_count_tab1 = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)), key="tab1_hover")

            if coords_for_plot is not None:
                # 1. 尝试加载/生成背景图
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
                    
                else:
                    with st.spinner("⏳ 正在绘制饼图背景..."):
                        # 定义新函数直接调用 utils 并处理保存
                        # ... (generate_and_save_background, 保持不变但是为了缩短代码这里略去具体定义，实际替换时需包含)
                        def generate_and_save_background(df, cds, size, save_dir):
                            img, bounds = utils.generate_clean_pie_chart(df, cds, size)
                            # 尝试保存到结果目录
                            try:
                                img_path = os.path.join(save_dir, "interactive_pie_background.png")
                                meta_path = os.path.join(save_dir, "interactive_pie_bounds.json")
                                img.save(img_path)
                                import json
                                with open(meta_path, 'w') as f:
                                    json.dump({'xlim': bounds[0], 'ylim': bounds[1]}, f)
                                return img, bounds, True 
                            except Exception as e:
                                return img, bounds, False

                        @st.cache_data(persist=True, show_spinner=False)
                        def get_cached_background(df, cds, size, save_path_key):
                            return utils.generate_clean_pie_chart(df, cds, size)
                        
                        # 1. 先计算
                        bg_img, (xlim, ylim) = get_cached_background(predict_df, coords_for_plot, None, result_dir)
                        
                        # 2. 检查并保存
                        target_img = os.path.join(result_dir, "interactive_pie_background.png")
                        if not os.path.exists(target_img):
                             try:
                                 bg_img.save(target_img)
                                 import json
                                 with open(os.path.join(result_dir, "interactive_pie_bounds.json"), 'w') as f:
                                     json.dump({'xlim': xlim, 'ylim': ylim}, f)
                             except:
                                 pass
                
                # 2. 准备交互数据 (使用 utils 封装函数)
                # 3. 颜色映射 (与 utils 中保持一致)
                cell_type_color_map = utils.get_color_map(predict_df.columns.tolist())

                fig = utils.generate_plotly_scatter(
                    coords_for_plot, predict_df, hover_count_tab1, 
                    bg_img, (xlim, ylim), cell_type_color_map
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                st.caption("💡 说明：此图背景为多色饼图，展示每个位置的细胞组成；鼠标悬停可查看具体比例数据。")
            else:
                 st.warning("缺少坐标数据，无法生成交互式图表。显示静态预览：")
                 pie_plot_path = os.path.join(result_dir, "predict_results_pie_plot.jpg")
                 if os.path.exists(pie_plot_path):
                     st.image(pie_plot_path, use_container_width=True)

        # --- Tab 2: 主要类型分布 (Dominant Scatter) ---
        with tabs[1]:
            st.subheader("主要类型分布 (优势细胞)")
            
            with st.expander("🛠️ 设置", expanded=False):
                hover_count = st.slider("悬停显示前 N 种细胞", 3, len(cell_types), min(6, len(cell_types)), key="tab2_hover")
            
            if coords_for_plot is not None:
                # 颜色映射
                unique_types = sorted(predict_df.columns.tolist())
                color_map = utils.get_color_map(unique_types)
                
                fig = utils.generate_dominant_scatter(
                    coords_for_plot, predict_df, hover_count, color_map
                )
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})
                st.caption("🖱️ 图例操作：单击显示/隐藏；双击独显当前类型。")
            else:
                st.warning("无法显示交互式图表（坐标数据不匹配）")
        
        # --- Tab 3: 整体比例统计 (Bar Chart) ---
        with tabs[2]:
            st.subheader("📊 整体比例统计")
            fig = utils.generate_proportion_bar(predict_df)
            st.plotly_chart(fig, use_container_width=True)

        # --- Tab 4: 单细胞类型热图 (Heatmap) ---
        with tabs[3]:
            selected_type = st.selectbox("🔬 选择要查看的细胞类型", cell_types, index=0)
            st.subheader(f"单细胞类型热图: {selected_type}")
            
            if coords_for_plot is not None:
                fig = utils.generate_heatmap(coords_for_plot, predict_df, selected_type)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})
            else:
                # 尝试显示静态图 fallback
                heatmap_path = os.path.join(result_dir, f"{selected_type}.jpg")
                if os.path.exists(heatmap_path):
                    st.image(heatmap_path, use_container_width=True)
                else:
                    st.warning("暂无该类型的坐标数据或静态图片。")
        
        # --- Tab 5: 详细数据表 (Table) ---
        with tabs[4]:
            st.subheader("详细数据表")
            st.dataframe(predict_df, use_container_width=True, height=400)
            
            csv = predict_df.to_csv()
            st.download_button(
                label="📥 下载 CSV",
                data=csv,
                file_name="predict_result.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
