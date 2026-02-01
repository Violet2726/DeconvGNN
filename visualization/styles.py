
import streamlit as st

def get_css():
    """返回应用自定义 CSS 样式字符串 - 2024 Premium Dark Theme。"""
    return """
    <style>
        /* ========== 0. 现代字体导入 ========== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* ========== 1. 基础设置 ========== */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* 禁止侧边栏/主区域动画 */
        section[data-testid="stSidebar"], .stMainBlockContainer {
            transition: none !important;
        }
        
        /* 图表容器最小宽度 & 边距优化 */
        .stMainBlockContainer {
            min-width: 1000px !important;
            max-width: 98% !important;     /* 扩大最大宽度 */
            padding-left: 1rem !important; /* 减少左边距 */
            padding-right: 1rem !important;/* 减少右边距 */
            padding-top: 1.5rem !important;  /* 顶部边距 */
            padding-bottom: 0rem !important; /* 底部边距 */
            overflow-x: auto !important;
        }
        
        /* 恢复 Header 但设为透明，确保展开按钮可见 */
        header[data-testid="stHeader"] {
            display: block !important;
            background: transparent !important;
            pointer-events: none !important; /* 让点击穿透 Header 背景 */
        }
        
        /* 恢复 Header 内按钮的点击交互 */
        header[data-testid="stHeader"] button, 
        [data-testid="stSidebarCollapsedControl"] {
            pointer-events: auto !important;
            color: rgba(255, 255, 255, 0.8) !important;
        }
        
        [data-testid="stPlotlyChart"] {
            width: 100% !important;
            min-width: 900px !important;
            height: 800px !important;
            overflow: hidden !important; 
        }
        
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
        
        /* ========== 2. 毛玻璃侧边栏 (Glassmorphism) ========== */
        section[data-testid="stSidebar"] > div:first-child {
            background: rgba(30, 30, 46, 0.75) !important;
            backdrop-filter: blur(16px) saturate(180%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        
        /* ========== 3. Logo 霓虹发光效果 ========== */
        .main-header {
            font-size: 1.6rem !important;
            font-weight: 700 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
            animation: glow-pulse 3s ease-in-out infinite alternate;
        }
        
        @keyframes glow-pulse {
            from { filter: drop-shadow(0 0 8px rgba(102, 126, 234, 0.4)); }
            to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.7)); }
        }
        
        .sub-header {
            font-size: 1rem;
            color: rgba(255, 255, 255, 0.7);
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        /* ========== 4. 指标卡片 (Metric Cards) ========== */
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(40, 40, 60, 0.6), rgba(30, 30, 50, 0.4)) !important;
            border-radius: 16px !important;
            padding: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
            backdrop-filter: blur(8px) !important;
            transition: transform 0.3s ease, box-shadow 0.3s ease !important;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-4px) !important;
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25) !important;
        }
        
        /* 指标数值渐变色 */
        div[data-testid="stMetricValue"] > div {
            background: linear-gradient(90deg, #00d4ff, #7b2ff7) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-weight: 700 !important;
        }
        
        /* 指标标签样式 */
        div[data-testid="stMetricLabel"] > div {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.85rem !important;
        }
        
        /* ========== 5. Tab 标签页 ========== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 30, 50, 0.3);
            border-radius: 12px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            background: transparent !important;
            border: none !important;
            color: rgba(255, 255, 255, 0.6) !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
            padding: 10px 16px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button:hover {
            color: #fff !important;
            background: rgba(102, 126, 234, 0.15) !important;
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            color: #fff !important;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3)) !important;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* Tab 底部指示条 */
        .stTabs [data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
            height: 3px !important;
            border-radius: 3px !important;
        }
        
        /* ========== 6. 按钮样式 ========== */
        /* 主要按钮 - 渐变 + 发光 */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: #fff !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            transition: all 0.3s ease !important;
            border-radius: 10px !important;
        }
        
        .stButton > button[kind="primary"]:hover,
        .stButton > button[data-testid="baseButton-primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6) !important;
        }
        
        /* 次要按钮 */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"] {
            background: transparent !important;
            border: 1px solid rgba(102, 126, 234, 0.5) !important;
            color: #667eea !important;
            transition: all 0.3s ease !important;
            border-radius: 10px !important;
        }
        
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="baseButton-secondary"]:hover {
            background: rgba(102, 126, 234, 0.1) !important;
            border-color: #667eea !important;
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* 侧边栏按钮优化 */
        div[data-testid="stHorizontalBlock"] button {
            white-space: nowrap;
            padding-left: 8px !important;
            padding-right: 8px !important;
        }
        
        /* ========== 7. 下拉菜单/选择器 ========== */
        div[data-baseweb="select"] > div {
            background: rgba(40, 40, 60, 0.6) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 10px !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"] > div:hover {
            border-color: rgba(102, 126, 234, 0.6) !important;
            box-shadow: 0 0 12px rgba(102, 126, 234, 0.15) !important;
        }
        
        /* ========== 8. 图表容器 ========== */
        [data-testid="stPlotlyChart"] {
            background: rgba(20, 20, 30, 0.4) !important;
            border-radius: 16px !important;
            padding: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }
        
        /* ========== 9. Expander 折叠面板 ========== */
        .streamlit-expanderHeader {
            background: rgba(40, 40, 60, 0.4) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.1) !important;
        }
        
        /* ========== 10. 输入框 ========== */
        .stTextInput > div > div > input {
            background: rgba(40, 40, 60, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            color: #fff !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
        }
        
        /* ========== 11. 滑块 ========== */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2) !important;
        }
        
        /* ========== 12. 分割线 ========== */
        hr {
            border: none !important;
            height: 1px !important;
            background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent) !important;
            margin: 1.5rem 0 !important;
        }
        
        /* ========== 13. 信息框美化 ========== */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
        }
        
        div[data-testid="stNotification"] {
            border-radius: 12px !important;
        }
        
    </style>
    """

def inject_custom_css():
    """向 Streamlit 应用注入自定义 CSS。"""
    st.markdown(get_css(), unsafe_allow_html=True)

