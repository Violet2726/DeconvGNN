"""
DeconvGNN-Vis 可视化系统入口
该模块负责构建基于 Streamlit 的 Web 界面，包括数据集管理、实时图表渲染及交互逻辑。
"""

import streamlit as st
import pandas as pd
import os
import logging
import re
import time
import math
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any


# --- 跨环境导入适配 (支持本地开发与 Streamlit Cloud) ---
try:
    import visualization.styles as styles
    import visualization.data_loader as data_loader
    import visualization.viz_utils as utils
except ImportError:
    import styles
    import data_loader
    import viz_utils as utils

# --- 页面全局配置 ---
st.set_page_config(
    page_title="DeconvGNN-Vis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed" # 初始收起侧边栏以展示欢迎页
)

def _get_env_int(key: str, default: int) -> int:
    try:
        value = int(os.getenv(key, default))
        return value if value > 0 else default
    except Exception:
        return default

def _get_logger() -> logging.Logger:
    logger = logging.getLogger("visualization.app")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(os.getenv("DECONV_VIS_LOG_LEVEL", "INFO").upper())
    return logger

logger = _get_logger()
MAX_UPLOAD_MB = _get_env_int("DECONV_VIS_MAX_UPLOAD_MB", 200)
DATASET_NAME_MAX_LEN = _get_env_int("DECONV_VIS_DATASET_NAME_MAX_LEN", 64)
MAX_PERF_RECORDS = _get_env_int("DECONV_VIS_MAX_PERF_RECORDS", 200)
SHOW_PERF_MONITOR_TAB = False

def normalize_dataset_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff\- ]+", "_", name).strip()
    cleaned = cleaned[:DATASET_NAME_MAX_LEN]
    return cleaned if cleaned else "dataset"

def ensure_unique_dataset_name(name: str, existing: Dict[str, str]) -> str:
    if name not in existing:
        return name
    counter = 1
    while f"{name}_{counter}" in existing:
        counter += 1
    return f"{name}_{counter}"

def render_messages(errors: List[str], warnings: List[str]) -> None:
    for err in errors:
        st.error(f"❌ {err}")
    for warn in warnings:
        st.warning(f"⚠️ {warn}")

def _init_perf_state() -> None:
    if "perf_metrics" not in st.session_state:
        st.session_state.perf_metrics = []
    if "perf_monitor_enabled" not in st.session_state:
        st.session_state.perf_monitor_enabled = True
    if "perf_mode" not in st.session_state:
        st.session_state.perf_mode = "高性能"
    if "parallel_load" not in st.session_state:
        st.session_state.parallel_load = True
    if "binary_cache" not in st.session_state:
        st.session_state.binary_cache = True
    if "prewarm_bg" not in st.session_state:
        st.session_state.prewarm_bg = True
    if "auto_prewarm" not in st.session_state:
        st.session_state.auto_prewarm = True
    if "summary_index" not in st.session_state:
        st.session_state.summary_index = True
    if "prewarm_pending" not in st.session_state:
        st.session_state.prewarm_pending = False
    if "prewarm_mode" not in st.session_state:
        st.session_state.prewarm_mode = "auto"
    if "prewarm_future" not in st.session_state:
        st.session_state.prewarm_future = None
    if "prewarm_notified" not in st.session_state:
        st.session_state.prewarm_notified = False
    if "mem_snapshot" not in st.session_state:
        st.session_state.mem_snapshot = None

def record_metric(label: str, duration_ms: float, extra: Optional[Dict[str, Any]] = None) -> None:
    if not st.session_state.get("perf_monitor_enabled"):
        return
    item = {
        "label": label,
        "duration_ms": round(duration_ms, 2),
        "mode": st.session_state.get("perf_mode", "标准"),
        "ts": pd.Timestamp.now().strftime("%H:%M:%S")
    }
    if extra:
        item.update(extra)
    st.session_state.perf_metrics.append(item)
    if len(st.session_state.perf_metrics) > MAX_PERF_RECORDS:
        st.session_state.perf_metrics = st.session_state.perf_metrics[-MAX_PERF_RECORDS:]

def run_timed(label: str, fn, extra: Optional[Dict[str, Any]] = None):
    start = time.perf_counter()
    result = fn()
    duration = (time.perf_counter() - start) * 1000
    record_metric(label, duration, extra)
    return result

def get_perf_mode() -> str:
    return st.session_state.get("perf_mode", "标准")


# 注入自定义样式（强制按钮不换行、隐藏默认菜单等）
styles.inject_custom_css()



def main():
    """
    应用核心入口函数，控制整体业务逻辑与界面流转。
    """
    _init_perf_state()
    if "figure_cache" not in st.session_state:
        st.session_state.figure_cache = {}
    
    # === 侧边栏区域：数据源管理 ===
    with st.sidebar:
        st.markdown('<p class="main-header">DeconvGNN-Vis<br>空间转录组反卷积<br>可视化系统</p>', unsafe_allow_html=True)
        st.divider()
        
        # 系统重置工具
        if st.button("⚡ 重置系统", type="secondary", use_container_width=True, help="清除所有缓存并重新加载应用"):
            st.cache_data.clear()
            st.rerun()
            
        st.header("数据集管理", help="""目标文件夹必须包含：  
            `predict_result.csv`  
            `coordinates.csv`""")
        
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
            if "last_dataset_name" not in st.session_state:
                st.session_state.last_dataset_name = selected_dataset_name
            elif selected_dataset_name != st.session_state.last_dataset_name:
                st.session_state.last_dataset_name = selected_dataset_name
                if st.session_state.auto_prewarm:
                    st.session_state.prewarm_pending = True
                    st.session_state.prewarm_mode = "auto"
                    st.session_state.prewarm_notified = False


        # 数据集操作工具栏 (删除与新增)
        col_del, col_add = st.columns(2)
        
        with col_del:
            if selected_dataset_name:
                if st.button("🗑️ 移除", type="secondary", use_container_width=True):
                    if st.session_state.data_sources.get(selected_dataset_name) == "__UPLOADED__":
                        if 'uploaded_data' in st.session_state:
                            del st.session_state.uploaded_data
                    del st.session_state.data_sources[selected_dataset_name]
                    # 重置选择器状态
                    if "dataset_selector" in st.session_state:
                        del st.session_state.dataset_selector
                    st.rerun()
            else:
                 st.button("🗑️ 移除", type="secondary", disabled=True, use_container_width=True)
 
        if 'rename_dialog_open' not in st.session_state:
            st.session_state.rename_dialog_open = False
            
        @st.dialog("重命名数据集")
        def rename_dialog(default_name, valid_path):
            new_name = st.text_input("显示名称", value=default_name)
            if st.button("确认添加", type="primary", use_container_width=True):
                if new_name.strip():
                    normalized_name = normalize_dataset_name(new_name)
                    normalized_name = ensure_unique_dataset_name(normalized_name, st.session_state.data_sources)
                    st.session_state.data_sources[normalized_name] = valid_path
                    st.session_state.dataset_selector = normalized_name
                    st.session_state.temp_import_path = "" # Clear path
                    st.session_state.rename_dialog_open = False # Close flag
                    
                    st.rerun()
                else:
                    st.error("名称不能为空")
        
        with col_add:
            is_cloud = utils.is_cloud_environment()
            
            if is_cloud:
                # Cloud: Toggle Button
                btn_label = "✖️ 取消" if st.session_state.show_import else "📂 导入"
                if st.button(btn_label, type="secondary", use_container_width=True):
                    st.session_state.show_import = not st.session_state.show_import
                    st.rerun()
            else:
                # Local: Direct Browse with Dialog
                if st.button("📂 导入", type="secondary", use_container_width=True):
                    folder = utils.open_folder_dialog()
                    if folder:
                         # Validate Path immediately
                        valid_path = None
                        if os.path.exists(os.path.join(folder, "predict_result.csv")):
                            valid_path = folder
                        elif os.path.exists(os.path.join(folder, "results", "predict_result.csv")):
                            valid_path = os.path.join(folder, "results")
                        
                        if valid_path:
                            st.session_state.temp_import_path = valid_path
                            st.session_state.rename_dialog_open = True
                            st.rerun()
                        else:
                            st.toast("❌ 目录无效：未找到 predict_result.csv", icon="🚫")

        # Trigger Dialog if flag is set (Local Only)
        if st.session_state.get('rename_dialog_open') and st.session_state.get('temp_import_path'):
            # Smart Naming: If the selected folder is 'results', use the parent folder name
            raw_basename = os.path.basename(st.session_state.temp_import_path)
            if raw_basename.lower() == "results":
                parent_name = os.path.basename(os.path.dirname(st.session_state.temp_import_path))
                base_name = parent_name
            else:
                base_name = raw_basename
            
            rename_dialog(normalize_dataset_name(base_name), st.session_state.temp_import_path)

        # Cloud Import Logic (Only visible if Cloud mode AND show_import is True)
        if is_cloud and st.session_state.show_import:
             with st.container():
                
                uploaded_files = st.file_uploader(
                    "上传数据文件",
                    type=["csv"],
                    accept_multiple_files=True,
                    help="请上传 predict_result.csv 和 coordinates.csv",
                    key="cloud_uploader",
                    label_visibility="collapsed"
                )
                
                if uploaded_files:
                    oversize_files = [
                        f.name for f in uploaded_files
                        if getattr(f, "size", 0) > MAX_UPLOAD_MB * 1024 * 1024
                    ]
                    if oversize_files:
                        st.error(f"上传文件过大，单文件上限 {MAX_UPLOAD_MB}MB: {', '.join(oversize_files)}")
                        return
                    file_names = [f.name.lower() for f in uploaded_files]
                    if any("predict" in name for name in file_names):
                        # Auto-generate default name: dataset_1, dataset_2, ...
                        counter = 1
                        while f"dataset_{counter}" in st.session_state.data_sources:
                            counter += 1
                        default_cloud_name = f"dataset_{counter}"
                        
                        new_name = st.text_input("数据集显示名称", value=default_cloud_name)
                        
                        def on_upload_confirm():
                            if new_name.strip():
                                pdf, cdf, errors, warnings = run_timed(
                                    "upload_parse",
                                    lambda: data_loader.load_from_uploaded_files(
                                        uploaded_files,
                                        use_parallel=st.session_state.parallel_load
                                    ),
                                    {"files": len(uploaded_files), "parallel": st.session_state.parallel_load}
                                )
                                render_messages(errors, warnings)
                                if errors:
                                    return
                                if pdf is not None:
                                    normalized_name = normalize_dataset_name(new_name)
                                    normalized_name = ensure_unique_dataset_name(normalized_name, st.session_state.data_sources)
                                    st.session_state.uploaded_data_cache = {
                                        'predict_df': pdf,
                                        'coords': cdf
                                    }
                                    st.session_state.data_sources[normalized_name] = "__UPLOADED__"
                                    st.session_state.dataset_selector = normalized_name
                                    st.session_state.show_import = False
                                else:
                                    st.toast("❌ 数据解析失败，请检查 CSV 格式", icon="❌")
                            else:
                                st.error("请输入名称")
                        
                        st.button("✅ 确认上传", type="primary", use_container_width=True, on_click=on_upload_confirm)
                    else:
                        st.warning("⚠️ 必需文件缺失：请务必上传 `predict_result.csv`")


    # === 主界面展示区 ===
    
    # 无数据场景：展示欢迎页与系统简介
    if result_dir is None:
        # 指向侧边栏的交互指引
        st.markdown('<div class="sidebar-hint"><i class="fa-solid fa-angles-left" style="font-size:3rem; color:#00f260; filter: drop-shadow(0 0 10px #00f260);"></i></div>', unsafe_allow_html=True)
        
        # 首页视觉渲染 (基于 Assets 图片与动态样式)
        banner_base64 = utils.get_base64_image_cached(str(utils.BANNER_PATH))
        banner_src = f"data:image/png;base64,{banner_base64}" if banner_base64 else ""
 
        st.markdown(styles.get_landing_page_html(banner_src), unsafe_allow_html=True)
        return
    

         
    prewarm_future = st.session_state.get("prewarm_future")
    if prewarm_future is not None and prewarm_future.done() and not st.session_state.prewarm_notified:
        prewarm_errors, prewarm_warnings = prewarm_future.result()
        st.session_state.prewarm_notified = True
        render_messages(prewarm_errors, prewarm_warnings)
        if prewarm_errors:
            st.toast("❌ 后台预热失败", icon="❌")
        else:
            st.toast("✅ 后台预热完成", icon="✅")

    if st.session_state.prewarm_pending:
        if result_dir and result_dir != "__UPLOADED__":
            if st.session_state.prewarm_mode == "manual":
                st.session_state.prewarm_pending = False
                with st.spinner("正在预热缓存..."):
                    prewarm_predict, prewarm_coords, prewarm_errors, prewarm_warnings = run_timed(
                        "prewarm_load",
                        lambda: data_loader.load_results(
                            result_dir,
                            use_parallel=st.session_state.parallel_load,
                            use_binary_cache=st.session_state.binary_cache
                        ),
                        {"parallel": st.session_state.parallel_load, "binary_cache": st.session_state.binary_cache}
                    )
                    render_messages(prewarm_errors, prewarm_warnings)
                    if prewarm_errors:
                        return
                    if st.session_state.prewarm_bg and prewarm_predict is not None and prewarm_coords is not None:
                        bg_cache_key = f"{selected_dataset_name}_bg_img"
                        if bg_cache_key not in st.session_state.figure_cache:
                            bg_img, (xlim, ylim) = run_timed(
                                "prewarm_bg",
                                lambda: utils.get_or_generate_pie_background(
                                    prewarm_predict,
                                    prewarm_coords,
                                    result_dir
                                ),
                                {"points": len(prewarm_predict)}
                            )
                            st.session_state.figure_cache[bg_cache_key] = {"img": bg_img, "xlim": xlim, "ylim": ylim}
                    st.toast("✅ 缓存预热完成", icon="✅")
            else:
                if prewarm_future is None or prewarm_future.done():
                    executor = ThreadPoolExecutor(max_workers=1)
                    st.session_state.prewarm_future = executor.submit(
                        data_loader.prewarm_cache,
                        result_dir,
                        st.session_state.parallel_load,
                        st.session_state.binary_cache
                    )
                    st.session_state.prewarm_pending = False
                    st.toast("⏳ 已开始后台预热", icon="⏳")
        else:
            st.session_state.prewarm_pending = False
            st.toast("⚠️ 仅本地数据支持预热缓存", icon="⚠️")

    # 有效数据场景：执行数据流加载
    if result_dir == "__UPLOADED__":
        # 云端部署加载逻辑：通过 Session State 恢复
        if 'uploaded_data_cache' in st.session_state:
            predict_df = st.session_state.uploaded_data_cache['predict_df']
            coords = st.session_state.uploaded_data_cache['coords']
            predict_df, coords, errors, warnings = run_timed(
                "data_validate_uploaded",
                lambda: data_loader.validate_dataset(predict_df, coords)
            )
            render_messages(errors, warnings)
            if errors:
                return
        else:
            st.error("❌ 会话过期：上传的数据已失效，请重新上传文件。")
            return
    else:
        # 本地开发加载逻辑：通过文件系统读取
        with st.spinner("正在加载数据集..."):
            predict_df, coords, errors, warnings = run_timed(
                "data_load_local",
                lambda: data_loader.load_results(
                    result_dir,
                    use_parallel=st.session_state.parallel_load,
                    use_binary_cache=st.session_state.binary_cache
                ),
                {"parallel": st.session_state.parallel_load, "binary_cache": st.session_state.binary_cache}
            )
        render_messages(errors, warnings)
        if errors:
            return
    
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
        
        perf_mode = get_perf_mode()
        use_lod = perf_mode == "高性能"
        tab_labels = [
            "🧩 空间组分图谱",
            "🔍 优势亚群分布",
            "📊 细胞比例概览",
            "🌡️ 单细胞热度图",
            "📑 原始数据详单"
        ]
        if SHOW_PERF_MONITOR_TAB:
            tab_labels.append("⚙️ 性能监控")
        tabs = st.tabs(tab_labels)
        
        with tabs[0]:
            coords_for_plot = coords
            if "tab1_hover_value" not in st.session_state:
                st.session_state.tab1_hover_value = min(6, len(cell_types))
            
            with st.expander("设置", expanded=False):
                st.slider(
                    "悬停详情数量",
                    3,
                    len(cell_types),
                    st.session_state.tab1_hover_value,
                    key="tab1_hover_value"
                )
            hover_count_tab1 = st.session_state.tab1_hover_value

            if coords_for_plot is not None:
                bg_cache_key = f"{cache_prefix}bg_img"
                if bg_cache_key in st.session_state.figure_cache:
                    cached_bg = st.session_state.figure_cache[bg_cache_key]
                    bg_img = cached_bg['img']
                    xlim, ylim = cached_bg['xlim'], cached_bg['ylim']
                else:
                    progress_bar = st.progress(0, text="🧪 正在通过并行管道计算空间饼图轨迹...")
                    def update_progress(pct, msg):
                        progress_bar.progress(pct, text=f"⏳ {msg}")
                    bg_img, (xlim, ylim) = run_timed(
                        "tab1_bg_generate",
                        lambda: utils.get_or_generate_pie_background(
                            predict_df,
                            coords_for_plot,
                            result_dir,
                            progress_callback=update_progress
                        ),
                        {"points": len(predict_df)}
                    )
                    progress_bar.empty()
                    st.session_state.figure_cache[bg_cache_key] = {'img': bg_img, 'xlim': xlim, 'ylim': ylim}

                plot_predict_df = predict_df
                plot_coords = coords_for_plot
                sampled = False
                if use_lod:
                    plot_predict_df, plot_coords, sampled = utils.apply_lod_sampling(predict_df, coords_for_plot)

                tab1_cache_key = f"{cache_prefix}tab1_{hover_count_tab1}_{'lod' if sampled else 'full'}"
                if tab1_cache_key not in st.session_state.figure_cache:
                    cell_type_color_map = utils.get_color_map_cached(tuple(plot_predict_df.columns.tolist()), plot_predict_df)
                    fig = run_timed(
                        "tab1_render",
                        lambda: utils.generate_plotly_scatter(
                            plot_coords,
                            plot_predict_df,
                            hover_count_tab1,
                            bg_img,
                            (xlim, ylim),
                            cell_type_color_map
                        ),
                        {"points": len(plot_predict_df), "sampled": sampled}
                    )
                    st.session_state.figure_cache[tab1_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab1_cache_key]

                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                st.caption("💡 视图说明：背景层展示各观测位点的多组分构成；悬停可探索亚细胞级占比详情。")
                if sampled:
                    st.caption("⚡ 当前为高性能采样视图")
            else:
                st.warning("⚠️ 坐标数据缺失或不兼容：无法生成空间拓扑图。")
                 
        with tabs[1]:
            if "tab2_hover_value" not in st.session_state:
                st.session_state.tab2_hover_value = min(6, len(cell_types))
            
            with st.expander("设置", expanded=False):
                st.slider(
                    "悬停详情数量",
                    3,
                    len(cell_types),
                    st.session_state.tab2_hover_value,
                    key="tab2_hover_value"
                )
            hover_count = st.session_state.tab2_hover_value

            if coords_for_plot is not None:
                plot_predict_df = predict_df
                plot_coords = coords_for_plot
                sampled = False
                if use_lod:
                    plot_predict_df, plot_coords, sampled = utils.apply_lod_sampling(predict_df, coords_for_plot)

                tab2_cache_key = f"{cache_prefix}tab2_{hover_count}_{'lod' if sampled else 'full'}"
                if tab2_cache_key not in st.session_state.figure_cache:
                    color_map = utils.get_color_map_cached(tuple(plot_predict_df.columns.tolist()), plot_predict_df)
                    fig = run_timed(
                        "tab2_render",
                        lambda: utils.generate_dominant_scatter(plot_coords, plot_predict_df, hover_count, color_map),
                        {"points": len(plot_predict_df), "sampled": sampled}
                    )
                    st.session_state.figure_cache[tab2_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab2_cache_key]

                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                st.caption("🖱️ 交互贴士：通过点击右侧图例可进行细胞类型筛选；双击可切换独显/全选模式。")
                if sampled:
                    st.caption("⚡ 当前为高性能采样视图")
            else:
                st.warning("⚠️ 数据异常：该数据集无法进行优势亚群聚类映射。")
        
        with tabs[2]:
            tab3_cache_key = f"{cache_prefix}tab3"

            if tab3_cache_key not in st.session_state.figure_cache:
                fig = run_timed(
                    "tab3_render",
                    lambda: utils.generate_proportion_bar(predict_df),
                    {"types": len(cell_types)}
                )
                st.session_state.figure_cache[tab3_cache_key] = fig
            else:
                fig = st.session_state.figure_cache[tab3_cache_key]

            st.plotly_chart(fig, use_container_width=True)

        with tabs[3]:
            if "tab4_type_value" not in st.session_state:
                st.session_state.tab4_type_value = cell_types[0]
            selected_index = 0
            if st.session_state.tab4_type_value in cell_types:
                selected_index = cell_types.index(st.session_state.tab4_type_value)
            st.selectbox("🔬 检索目标细胞亚群", cell_types, index=selected_index, key="tab4_type_value")
            selected_type = st.session_state.tab4_type_value

            if coords_for_plot is not None:
                plot_predict_df = predict_df
                plot_coords = coords_for_plot
                sampled = False
                if use_lod:
                    plot_predict_df, plot_coords, sampled = utils.apply_lod_sampling(predict_df, coords_for_plot)

                tab4_cache_key = f"{cache_prefix}tab4_{selected_type}_{'lod' if sampled else 'full'}"
                if tab4_cache_key not in st.session_state.figure_cache:
                    fig = run_timed(
                        "tab4_render",
                        lambda: utils.generate_heatmap(plot_coords, plot_predict_df, selected_type),
                        {"points": len(plot_predict_df), "sampled": sampled}
                    )
                    st.session_state.figure_cache[tab4_cache_key] = fig
                else:
                    fig = st.session_state.figure_cache[tab4_cache_key]

                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False, 'responsive': True})
                if sampled:
                    st.caption("⚡ 当前为高性能采样视图")
            else:
                st.warning("⚠️ 提示：缺少该样本的空间坐标。")

        with tabs[4]:
            st.markdown("#### 📑 反卷积预测原始指标矩阵")
            st.dataframe(predict_df, use_container_width=True, height=500)

        if SHOW_PERF_MONITOR_TAB:
            with tabs[-1]:
                st.markdown("#### ⚙️ 性能监控")
                with st.expander("性能设置", expanded=False):
                    if st.button("清空性能记录", type="secondary", use_container_width=True):
                        st.session_state.perf_metrics = []
                        st.session_state.mem_snapshot = None
                    if st.button("预热缓存", type="secondary", use_container_width=True):
                        st.session_state.prewarm_pending = True
                        st.session_state.prewarm_mode = "manual"
                        st.session_state.prewarm_notified = False
                    prewarm_future = st.session_state.get("prewarm_future")
                    if prewarm_future is not None and not prewarm_future.done():
                        st.caption("后台预热中…")
                if st.session_state.get("perf_monitor_enabled"):
                    if st.button("采集内存快照", type="secondary", use_container_width=False):
                        if not tracemalloc.is_tracing():
                            tracemalloc.start()
                        current, peak = tracemalloc.get_traced_memory()
                        st.session_state.mem_snapshot = {
                            "current_mb": round(current / 1024 / 1024, 2),
                            "peak_mb": round(peak / 1024 / 1024, 2)
                        }

                    metrics_df = pd.DataFrame(st.session_state.perf_metrics)
                    if not metrics_df.empty:
                        st.dataframe(metrics_df.tail(200), use_container_width=True, height=280)
                        summary = (
                            metrics_df
                            .groupby(["mode", "label"])["duration_ms"]
                            .agg(
                                count="count",
                                avg_ms="mean",
                                p95_ms=lambda s: s.quantile(0.95)
                            )
                            .reset_index()
                        )
                        st.dataframe(summary, use_container_width=True, height=260)
                    if st.session_state.mem_snapshot:
                        st.metric("当前内存(MB)", st.session_state.mem_snapshot["current_mb"])
                        st.metric("峰值内存(MB)", st.session_state.mem_snapshot["peak_mb"])
                else:
                    st.info("启用性能监控后可查看渲染耗时与内存快照")

if __name__ == "__main__":
    main()
