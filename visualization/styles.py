
import streamlit as st

def get_css():
    """返回应用自定义 CSS 样式字符串。"""
    return """
    <style>
        /* 隐藏 Streamlit 默认菜单和水印，保留侧边栏按钮 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 禁止侧边栏拖拽调整宽度，但保留收起/展开功能 */
        [data-testid="stSidebarResizeHandle"] {
            display: none !important;
            pointer-events: none !important;
            cursor: default !important;
        }
        
        /* 侧边栏边缘区域鼠标光标恢复默认 */
        section[data-testid="stSidebar"]::after,
        section[data-testid="stSidebar"] {
            cursor: default !important;
        }
        
        /* 覆盖所有可能的拖拽光标样式 */
        [data-testid="stSidebar"] *,
        .stSidebar * {
            cursor: default;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] a,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] input {
            cursor: pointer !important;
        }
        
        /* 侧边栏展开时固定宽度 */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 300px !important;
            min-width: 300px !important;
            max-width: 300px !important;
        }
        
        /* 侧边栏收起时确保主页面自动扩展 */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0px !important;
            min-width: 0px !important;
        }
        
        /* 主页面区域自动适应 */
        .stMainBlockContainer {
            transition: margin-left 0.3s ease;
        }
        
        .main-header {
            font-size: 1.45rem !important;
            font-weight: bold;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #666;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        
        /* 侧边栏按钮样式优化：强制不换行，保持紧凑 */
        div[data-testid="stHorizontalBlock"] button {
            white-space: nowrap;
            padding-left: 5px !important;
            padding-right: 5px !important;
        }
    </style>
    """

def inject_custom_css():
    """向 Streamlit 应用注入自定义 CSS。"""
    st.markdown(get_css(), unsafe_allow_html=True)
