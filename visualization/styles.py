# -*- coding: utf-8 -*-
"""
DeconvGNN-Vis 样式与首页 HTML 生成模块。

该文件集中维护 Streamlit 页面注入的 CSS，以及无数据状态下的落地页 HTML。
把样式从 `app.py` 中拆出，可以让页面逻辑与视觉实现解耦，后续调整主题、
动画或品牌元素时不影响数据加载与图表逻辑。
"""
import streamlit as st
import os
import base64

def _get_stardust_b64():
    """
    读取本地星尘纹理并转换为 Base64 URL。

    本地资源可避免部署环境访问外链失败；若资源缺失，则回退到公共纹理 URL，
    保证页面仍能正常渲染。
    """
    # 优先读取本地纹理资源，减少外部网络依赖。
    path = os.path.join(os.path.dirname(__file__), "assets", "stardust.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    # 本地资源缺失时回退到在线纹理。
    return "https://www.transparenttextures.com/patterns/stardust.png"

def get_css():
    """
    汇集应用自定义样式表。

    样式包含全局字体、暗色背景、侧边栏布局、按钮、指标卡片、Plotly 容器、
    首页动画等 Streamlit 覆盖规则。返回值会由 `inject_custom_css` 注入页面。
    """
    css_template = """
    <style>
        /* ==========================================================================
           0. 全局设置与字体引入
           统一字体、颜色变量和玻璃态基础参数，后续组件尽量复用变量。
           ========================================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #00f260;
            --bg-dark: #0e1117;
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-bg: rgba(20, 20, 35, 0.6);
            --neon-glow: 0 0 10px rgba(102, 126, 234, 0.6), 0 0 20px rgba(102, 126, 234, 0.4);
        }

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif !important;
            scroll-behavior: smooth;
        }

        /* 隐藏 Streamlit 默认装饰，让页面更像独立应用而不是脚本面板。 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stDecoration"] { display: none; }

        /* ==========================================================================
           1. 布局重构
           ========================================================================== */
        
        /* 1.1 主内容容器：沉浸式顶部与满宽布局。 */
        .stMainBlockContainer {
            padding-top: 3rem !important;   /* 顶部保留适度留白 */
            padding-bottom: 2rem !important;
            margin-top: 0 !important;       /* 自然排列 */
            max-width: 100% !important;     /* 占满全宽 */
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }

        /* 1.2 顶部 Header：完全透明，避免遮挡首页视觉背景。 */
        header {
            background: transparent !important;
            backdrop-filter: none !important;
        }
        
        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
            height: 3rem !important;
            pointer-events: none !important; /* 让点击穿透 */
        }
        
        /* 恢复 Header 内按钮交互。 */
        header[data-testid="stHeader"] button, 
        [data-testid="stSidebarCollapsedControl"],
        .stDeployButton {
            pointer-events: auto !important;
            color: rgba(255, 255, 255, 0.5) !important;
        }

        /* ==========================================================================
           2. 侧边栏体系
           ========================================================================== */
        
        /* 2.1 样式重置：完全透明，无边框、无阴影，与主背景融合。 */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: transparent !important;
            background: transparent !important;
            backdrop-filter: none !important;
            border-right: none !important;
            box-shadow: none !important;
        }

        /* 2.2 尺寸锁定：限制宽度为 300px，收起时归零。 */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 300px !important;
            max-width: 300px !important;
            width: 300px !important;
        }
        
        section[data-testid="stSidebar"][aria-expanded="false"] {
             min-width: 0 !important;
             max-width: 0 !important;
             width: 0 !important;
        }

        /* 2.3 交互限制：禁止拖拽，避免 Streamlit 默认侧栏宽度破坏布局。 */
        [data-testid="stSidebarResizeHandle"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* 2.4 光标管理：全局默认，仅组件可点。 */
        /* 覆盖侧边栏所有区域的光标为默认，避免非交互区域误导用户。 */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"]::before,
        section[data-testid="stSidebar"]::after {
            cursor: default !important;
            resize: none !important;
        }

        /* 恢复交互组件的手型光标。 */
        section[data-testid="stSidebar"] button,
        section[data-testid="stSidebar"] a,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] [role="button"] {
            cursor: pointer !important;
        }

        /* 侧边栏标题：霓虹流光特效。 */
        .main-header {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #fff 0%, #a5b4fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
            position: relative;
            text-shadow: 0 0 30px rgba(102, 126, 234, 0.5);
            letter-spacing: -0.5px;
        }

        /* ==========================================================================
           3. 全局背景
           ========================================================================== */
        .stApp {
            background-color: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(102, 126, 234, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(118, 75, 162, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(0, 242, 96, 0.1) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(5, 117, 230, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            background-size: 100% 100%;
        }
        
        /* 叠加星尘纹理。 */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: url("__STARDUST_IMAGE__");
            opacity: 0;
            pointer-events: none;
            z-index: 0;
        }

        /* ==========================================================================
           4. 组件样式 (UI Components)
           ========================================================================== */
        
        /* 4.1 按钮 (Buttons) */
        /* Primary - 镭射渐变 */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            letter-spacing: 0.5px !important;
            padding: 0.6rem 1.2rem !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }
        
        .stButton > button[kind="primary"]::after {
            content: '';
            position: absolute;
            top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: 0.5s;
        }
        
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(118, 75, 162, 0.6), 0 0 15px rgba(102, 126, 234, 0.5) !important;
        }
        
        .stButton > button[kind="primary"]:hover::after {
            left: 100%;
        }

        /* Secondary - 幽灵边框 */
        /* ================ 侧边栏按钮专属样式 (优先级最高) ================ */
        section[data-testid="stSidebar"] .stButton button:not([kind="primary"]) {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(4px);
            transition: all 0.3s ease !important;
            position: relative;
            z-index: 10002 !important;
            pointer-events: auto !important;
        }

        /* 侧边栏按钮悬停态 - 强制生效 */
        section[data-testid="stSidebar"] .stButton button:not([kind="primary"]):hover,
        section[data-testid="stSidebar"] .stButton button:not([kind="primary"]):active {
            border-color: #667eea !important;
            background: rgba(102, 126, 234, 0.1) !important;
            color: #fff !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
            text-shadow: 0 0 8px rgba(102, 126, 234, 0.6) !important;
            transform: translateY(-1px) !important;
        }
        
        /* 侧边栏按钮内的文字变色 */
        section[data-testid="stSidebar"] .stButton button:not([kind="primary"]):hover p {
            color: #fff !important;
        }

        /* ================ 全局次要按钮 (Secondary Global) ================ */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"] {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(4px);
            transition: all 0.3s ease !important;
        }
        
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="baseButton-secondary"]:hover {
            border-color: #667eea !important;
            background: rgba(102, 126, 234, 0.1) !important;
            color: #fff !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
            text-shadow: 0 0 8px rgba(102, 126, 234, 0.6) !important;
            transform: translateY(-1px) !important;
        }

        /* 4.2 数据指标卡 (Metrics)*/
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-left: 3px solid var(--primary-color) !important; /* 左侧强调色 */
            border-radius: 12px !important;
            padding: 1.5rem !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
            position: relative;
            overflow: hidden;
        }
        
        div[data-testid="stMetric"]::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at 80% 20%, rgba(102, 126, 234, 0.1), transparent 40%);
            pointer-events: none;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-5px) scale(1.01) !important;
            border-color: rgba(102, 126, 234, 0.5) !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.15) !important;
            background: linear-gradient(145deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.02) 100%) !important;
        }
        
        div[data-testid="stMetricLabel"] > div {
            font-family: 'JetBrains Mono', monospace !important;
            color: rgba(255, 255, 255, 0.7) !important; /* 提高标签清晰度 */
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem !important;
        }
        
        div[data-testid="stMetricValue"] > div {
            font-size: 2.2rem !important; /* 加大数值 */
            font-weight: 700 !important;
            background: linear-gradient(90deg, #fff 0%, #a5b4fc 100%); /* 数值渐变 */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 2px 10px rgba(102, 126, 234, 0.5));
        }

        /* 4.3 输入控件 (Inputs & Selects) */
        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input {
            background: rgba(40, 40, 60, 0.6) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 10px !important;
            color: #fff !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-baseweb="select"] > div:hover,
        .stTextInput > div > div > input:focus {
            border-color: rgba(102, 126, 234, 0.8) !important;
            box-shadow: 0 0 12px rgba(102, 126, 234, 0.2) !important;
        }

        /* 4.4 Tab 标签页 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: transparent !important; /* 移除容器背景 */
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: rgba(255, 255, 255, 0.6) !important;
            border-radius: 8px !important; /* 矩形圆角 */
            padding: 8px 24px !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }
        
        .stTabs [data-baseweb="tab-list"] button:hover {
            background: rgba(102, 126, 234, 0.2) !important;
            border-color: rgba(102, 126, 234, 0.5) !important;
            color: #fff !important;
            transform: translateY(-2px);
            box-shadow: 0 0 15px rgba(102, 126, 234, 0.4) !important; /* 霓虹光晕 */
            text-shadow: 0 0 8px rgba(102, 126, 234, 0.6) !important; /* 文字发光 */
        }
        
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border-color: transparent !important;
            color: #fff !important;
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.5) !important;
        }
        
        /* 隐藏默认的下划线 */
        .stTabs [data-baseweb="tab-highlight"] {
            display: none !important;
        }

        /* 4.5 Expander 折叠面板 */
        .streamlit-expanderHeader {
            background: rgba(40, 40, 60, 0.4) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* ==========================================================================
           5. 落地页 / Hero Section
           ========================================================================== */
        
        .landing-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
        }

        .banner-container {
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 40px 100px rgba(0,0,0,0.8);
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.05);
            height: 480px; /* 电影级高度 */
        }
        
        .banner-image {
            width: 100% !important;
            height: 100% !important;
            object-fit: cover !important;
            object-position: center center !important;
            transform: scale(1.02);
            filter: brightness(0.48) contrast(1.1) saturate(1.1);
            transition: none;
            min-width: 100%;
            min-height: 100%;
            display: block;
            animation: cinematic-zoom 20s ease-in-out infinite alternate;
        }

        @keyframes cinematic-zoom {
            0% { transform: scale(1.02); }
            100% { transform: scale(1.15); }
        }
        
        /* 移除两侧强烈的黑边遮罩，仅保留底部文字衬托 */
        .banner-container::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(to bottom, transparent 50%, rgba(14, 17, 23, 0.6) 100%);
            z-index: 1;
        }
        
        /* 扫描线动画 */
        .banner-container::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; height: 100%;
            background: linear-gradient(to bottom, transparent 50%, rgba(0, 242, 96, 0.03) 51%, transparent 52%);
            background-size: 100% 8px;
            z-index: 2;
            pointer-events: none;
        }

        /* Hero 文字内容容器 */
        .hero-section {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -40%);
            z-index: 10;
            text-align: center;
            width: 100%;
            padding: 0 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .hero-tagline {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: rgba(0, 242, 96, 0.8);
            letter-spacing: 0.4em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            opacity: 0.8;
            border-bottom: 1px solid rgba(0, 242, 96, 0.3);
            padding-bottom: 5px;
            display: inline-block;
        }

        .hero-title-main {
            font-size: 6rem !important;
            font-weight: 800 !important;
            letter-spacing: -2px;
            line-height: 1.1;
            margin: 0.5rem 0 1.5rem 0;
            background: linear-gradient(180deg, #ffffff 10%, #a5b4fc 60%, #667eea 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.8));
            position: relative;
        }
        
        .hero-title-main::before, .hero-title-main::after {
            content: '+';
            font-size: 1.5rem;
            color: rgba(255,255,255,0.3);
            vertical-align: middle;
            margin: 0 20px;
            font-weight: 300;
        }
        
        .hero-subtitle {
            background: rgba(14, 17, 23, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            padding: 10px 30px;
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            letter-spacing: 1px;
            font-weight: 400;
            display: inline-flex;
            align-items: center;
            gap: 12px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle::before {
            content: '●';
            color: #00f260;
            font-size: 0.8rem;
            animation: blink 2s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; text-shadow: 0 0 10px #00f260; }
            50% { opacity: 0.4; text-shadow: none; }
        }

        /* 功能卡片布局 */
        .features-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin-top: 4rem;
        }

        .bio-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 2.5rem 2rem;
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            position: relative;
            overflow: hidden;
        }

        .bio-card:hover {
            background: rgba(102, 126, 234, 0.06);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateY(-12px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }

        .bio-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            opacity: 0;
            transition: opacity 0.4s;
        }
        .bio-card:hover::before { opacity: 1; }

        .card-icon-large {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            display: block;
        }
        .card-title-main {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #fff;
        }
        .card-description {
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.5);
            line-height: 1.6;
        }
        
        /* 侧边栏引导箭头特效 */
        .sidebar-hint {
            position: fixed; top: 50%; left: 24px; transform: translateY(-50%);
            animation: float 3s ease-in-out infinite;
            z-index: 100;
        }
        @keyframes float {
            0%, 100% { transform: translate(0, -50%); opacity: 0.6; }
            50% { transform: translate(10px, -50%); opacity: 1; text-shadow: 0 0 15px #00f260; }
        }
        
        /* 快速启动指南 */
        .step-guide {
            margin-top: 5rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 3rem;
            border: 1px dashed rgba(255, 255, 255, 0.1);
            text-align: center;
            backdrop-filter: blur(5px);
        }

        .step-title {
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 1.3rem;
            margin-bottom: 1rem;
        }
        
        .step-desc {
            color: rgba(255,255,255,0.6); 
            margin: 0;
            font-size: 1rem;
        }

        /* ==========================================================================
           6. 交互式数据总览 (Interactive Data Overview)
           ========================================================================== */
        .data-overview-section {
            margin-top: 4rem;
        }

        .section-header {
            margin-bottom: 0.5rem;
        }

        .section-title {
            font-size: 1.6rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .section-subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.95rem;
            margin-bottom: 2rem;
        }

        .overview-cards {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
        }

        .overview-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.5rem 1.2rem;
            text-align: left;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }

        .overview-card::before {
            content: '';
            position: absolute;
            top: 0; right: 0;
            width: 60px; height: 60px;
            background: radial-gradient(circle, rgba(102, 126, 234, 0.15), transparent 70%);
            pointer-events: none;
        }

        .overview-card:hover {
            background: rgba(102, 126, 234, 0.08);
            border-color: rgba(102, 126, 234, 0.4);
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }

        .overview-card-icon {
            font-size: 1.2rem;
            margin-bottom: 0.8rem;
            display: block;
        }

        .overview-card-label {
            font-size: 0.8rem;
            color: rgba(255, 255, 255, 0.5);
            margin-bottom: 0.3rem;
            font-weight: 500;
        }

        .overview-card-title {
            font-size: 1.1rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* ==========================================================================
           7. 技术栈展示条 (Tech Stack Marquee)
           ========================================================================== */
        .tech-stack-section {
            margin-top: 4rem;
            padding: 2rem 0;
        }

        .tech-stack-header {
            margin-bottom: 1.5rem;
        }

        .tech-stack-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 0.3rem;
        }

        .tech-stack-subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9rem;
        }

        .marquee-container {
            overflow: hidden;
            position: relative;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* 左右渐隐遮罩 */
        .marquee-container::before,
        .marquee-container::after {
            content: '';
            position: absolute;
            top: 0; bottom: 0;
            width: 80px;
            z-index: 2;
            pointer-events: none;
        }

        .marquee-container::before {
            left: 0;
            background: linear-gradient(to right, rgba(14, 17, 23, 1), transparent);
        }

        .marquee-container::after {
            right: 0;
            background: linear-gradient(to left, rgba(14, 17, 23, 1), transparent);
        }

        .marquee-track {
            display: flex;
            gap: 3rem;
            animation: marquee-scroll 25s linear infinite;
            width: max-content;
        }

        @keyframes marquee-scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }

        .tech-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            min-width: 80px;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }

        .tech-item:hover {
            opacity: 1;
        }

        .tech-icon {
            width: 48px;
            height: 48px;
            object-fit: contain;
            filter: grayscale(20%);
            transition: filter 0.3s ease, transform 0.3s ease;
        }

        .tech-item:hover .tech-icon {
            filter: grayscale(0%) drop-shadow(0 0 8px rgba(102, 126, 234, 0.5));
            transform: scale(1.1);
        }

        .tech-name {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.6);
            font-weight: 500;
        }
    </style>
    """
    return css_template.replace("__STARDUST_IMAGE__", _get_stardust_b64())


def inject_custom_css():
    """
    将封装的 CSS 样式表注入当前 Streamlit 会话上下文。

    Streamlit 不提供全局主题运行时 API，因此这里通过 `st.markdown`
    注入 `<style>` 标签统一覆盖组件样式。
    """
    st.markdown(get_css(), unsafe_allow_html=True)

def get_landing_page_html(banner_src):
    """
    构造系统初始化引导页面的完整 HTML 结构（含内嵌 CSS）。

    参数:
        banner_src: Base64 图片 URL 或普通图片 URL，用作首页主视觉背景。

    返回:
        str: 可传入 `st.components.v1.html()` 的 HTML 字符串。
    """
    # 落地页专用 CSS。首页由 components.html 渲染，不直接继承外层 Streamlit DOM。
    landing_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: transparent;
            color: #fff;
        }
        
        .landing-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .banner-container {
            border-radius: 20px;
            overflow: hidden;
            position: relative;
            box-shadow: 0 40px 100px rgba(0,0,0,0.8);
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.05);
            height: 480px;
        }
        
        .banner-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center center;
            transform: scale(1.02);
            filter: brightness(0.48) contrast(1.1) saturate(1.1);
            animation: cinematic-zoom 20s ease-in-out infinite alternate;
        }

        @keyframes cinematic-zoom {
            0% { transform: scale(1.02); }
            100% { transform: scale(1.15); }
        }
        
        .banner-container::after {
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(to bottom, transparent 50%, rgba(14, 17, 23, 0.6) 100%);
            z-index: 1;
        }

        .hero-section {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -40%);
            z-index: 10;
            text-align: center;
            width: 100%;
            padding: 0 2rem;
        }

        .hero-tagline {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: rgba(0, 242, 96, 0.8);
            letter-spacing: 0.4em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid rgba(0, 242, 96, 0.3);
            padding-bottom: 5px;
            display: inline-block;
        }

        .hero-title-main {
            font-size: 5rem;
            font-weight: 800;
            letter-spacing: -2px;
            line-height: 1.1;
            margin: 0.5rem 0 1.5rem 0;
            background: linear-gradient(180deg, #ffffff 10%, #a5b4fc 60%, #667eea 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.8));
        }
        
        .hero-subtitle {
            background: rgba(14, 17, 23, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 50px;
            padding: 10px 30px;
            color: rgba(255,255,255,0.9);
            font-size: 1.1rem;
            display: inline-flex;
            align-items: center;
            gap: 12px;
        }
        
        .hero-subtitle::before {
            content: '●';
            color: #00f260;
            font-size: 0.8rem;
            animation: blink 2s infinite;
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; text-shadow: 0 0 10px #00f260; }
            50% { opacity: 0.4; text-shadow: none; }
        }

        .hero-cta {
            margin-top: 1.5rem;
        }

        .hero-cta-btn {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            padding: 14px 32px;
            border-radius: 50px;
            font-size: 1rem;
            font-weight: 600;
            text-decoration: none;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .hero-cta-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }

        .trust-badges {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 2rem;
            padding: 1.5rem 0;
            border-top: 1px solid rgba(255,255,255,0.05);
        }

        .trust-badge {
            text-align: center;
        }

        .trust-badge-value {
            font-size: 3.2rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .trust-badge-label {
            font-size: 1.1rem;
            color: #94a3b8; /* 冷蓝灰，低调哑光 */
            margin-top: 0.3rem;
            font-weight: 400;
        }

        .use-cases-section {
            margin-top: 3rem;
        }

        .use-cases-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            max-width: 800px;
            margin: 0 auto;
        }

        .use-case-tag {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.8rem 1.5rem;
            border-radius: 50px;
            font-size: 1rem;
            color: rgba(255,255,255,0.7);
            transition: all 0.3s ease;
        }

        .use-case-tag:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.3);
            color: #fff;
        }

        .features-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
            margin-top: 2rem;
        }

        .bio-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 2rem;
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            position: relative;
            overflow: hidden;
        }

        .bio-card:hover {
            background: rgba(102, 126, 234, 0.06);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }

        .card-title-main {
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-title-main::before {
            content: '';
            width: 4px;
            height: 1.4rem;
            background: linear-gradient(180deg, #00f260, #0575E6);
            border-radius: 2px;
        }

        .card-description {
            font-size: 1.1rem;
            color: #94a3b8; /* 冷蓝灰，低调哑光 */
            line-height: 1.7;
            letter-spacing: 0.02em;
            font-weight: 300;
        }

        .data-overview-section {
            margin-top: 3rem;
        }

        .section-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .section-subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.9rem;
            margin-bottom: 1.5rem;
        }

        .overview-cards {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 16px;
        }

        .overview-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.2rem 1rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }

        .overview-card:hover {
            background: rgba(102, 126, 234, 0.1);
            border-color: rgba(102, 126, 234, 0.4);
            transform: translateY(-4px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
        }

        .overview-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 0.8rem;
        }

        .blink-dot {
            width: 8px;
            height: 8px;
            background-color: #00f260;
            border-radius: 50%;
            box-shadow: 0 0 10px #00f260;
            animation: blink-dot-anim 2s infinite;
            flex-shrink: 0;
        }

        @keyframes blink-dot-anim {
            0% { opacity: 0.4; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1.2); box-shadow: 0 0 15px #00f260; }
            100% { opacity: 0.4; transform: scale(0.8); }
        }

        .overview-card-title {
            font-size: 1.3rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }

        .overview-module-type {
            font-size: 0.9rem;
            color: #94a3b8;
            font-weight: 400;
            display: block;
            margin-left: 18px; /* Indent to align with text above (dot width + gap) */
        }

        .tech-stack-section {
            margin-top: 3rem;
        }

        .tech-stack-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(90deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
        }

        .tech-stack-subtitle {
            color: rgba(255, 255, 255, 0.5);
            font-size: 0.85rem;
            margin-bottom: 1rem;
        }

        .marquee-container {
            overflow: hidden;
            position: relative;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            padding: 1.2rem 0;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .marquee-container::before,
        .marquee-container::after {
            content: '';
            position: absolute;
            top: 0; bottom: 0;
            width: 60px;
            z-index: 2;
            pointer-events: none;
        }

        .marquee-container::before {
            left: 0;
            background: linear-gradient(to right, rgba(14, 17, 23, 1), transparent);
        }

        .marquee-container::after {
            right: 0;
            background: linear-gradient(to left, rgba(14, 17, 23, 1), transparent);
        }

        .marquee-track {
            display: flex;
            gap: 2.5rem;
            animation: marquee-scroll 30s linear infinite;
            width: max-content;
        }

        @keyframes marquee-scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-25%); }
        }

        .tech-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.4rem;
            min-width: 70px;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }

        .tech-item:hover {
            opacity: 1;
        }

        .tech-icon {
            width: 40px;
            height: 40px;
            object-fit: contain;
        }

        .tech-name {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.6);
        }

        .step-guide {
            margin-top: 3rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 2rem;
            border: 1px dashed rgba(255, 255, 255, 0.1);
            text-align: center;
        }

        .step-title {
            background: linear-gradient(90deg, #00f260, #0575E6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 1.2rem;
            margin-bottom: 0.8rem;
        }
        
        .step-desc {
            color: rgba(255,255,255,0.6); 
            font-size: 0.95rem;
        }
    </style>
    """
    # 技术栈图标使用 CDN 图标；它们仅影响首页装饰，加载失败不影响核心功能。
    tech_stack = [
        ("Streamlit", "https://streamlit.io/images/brand/streamlit-mark-color.svg"),
        ("PyTorch", "https://pytorch.org/assets/images/pytorch-logo.png"),
        ("PyG", "https://raw.githubusercontent.com/pyg-team/pyg_sphinx_theme/master/pyg_sphinx_theme/static/img/pyg_logo.png"),
        ("Pandas", "https://pandas.pydata.org/static/img/pandas_mark.svg"),
        ("NumPy", "https://numpy.org/images/logo.svg"),
        ("SciPy", "https://scipy.org/images/logo.svg"),
        ("Scikit-learn", "https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"),
        ("Plotly", "https://images.plot.ly/logo/new-branding/plotly-logomark.png"),
        ("Matplotlib", "https://matplotlib.org/stable/_static/logo_dark.svg"),
        ("Seaborn", "https://seaborn.pydata.org/_static/logo-wide-lightbg.svg"),
        ("Scanpy", "https://scanpy.readthedocs.io/en/stable/_static/Scanpy_Logo_BrightFG.svg"),
        ("AnnData", "https://anndata.readthedocs.io/en/latest/_static/anndata_schema.svg"),
        ("NetworkX", "https://networkx.org/_static/networkx_logo.svg"),
    ]
    
    # 生成技术栈 HTML。复制 4 份用于无缝滚动，避免动画末尾出现空白断档。
    tech_items_html = ""
    for name, icon_url in tech_stack * 4:  # 复制 4 份确保无缝循环。
        tech_items_html += f'''
            <div class="tech-item">
                <img src="{icon_url}" alt="{name}" class="tech-icon">
                <span class="tech-name">{name}</span>
            </div>
        '''
    
    return landing_css + f"""
<div class="landing-wrapper">
    <div class="banner-container">
        <img src="{banner_src}" class="banner-image">
        <div class="banner-overlay"></div>
        <div class="hero-section">
            <div class="hero-tagline">Spatial Transcriptomics Analysis</div>
            <h1 class="hero-title-main">DeconvGNN Vis</h1>
            <p class="hero-subtitle">基于深度图神经网络的高性能空间转录组反卷积分析平台</p>
        </div>
    </div>
    
    <div class="trust-badges">
        <div class="trust-badge">
            <div class="trust-badge-value">50,000+</div>
            <div class="trust-badge-label">空间位点支持</div>
        </div>
        <div class="trust-badge">
            <div class="trust-badge-value">30+</div>
            <div class="trust-badge-label">细胞类型识别</div>
        </div>
        <div class="trust-badge">
            <div class="trust-badge-value">GNN</div>
            <div class="trust-badge-label">图神经网络驱动</div>
        </div>
        <div class="trust-badge">
            <div class="trust-badge-value">Open</div>
            <div class="trust-badge-label">开源研究工具</div>
        </div>
    </div>

    <div class="features-container">
        <div class="bio-card">
            <div class="card-title-main">WebGL 2.0 加速</div>
            <p class="card-description">底层采用 GPU 加速渲染引擎，支持数万级空间位点实时交互，缩放平移顺滑无阻。</p>
        </div>
        <div class="bio-card">
            <div class="card-title-main">超分辨率反卷积</div>
            <p class="card-description">集成先进的 GNN 架构，提供亚细胞级的组分还原，精准锁定每一个空间位点的细胞构成。</p>
        </div>
        <div class="bio-card">
            <div class="card-title-main">智能显存缓存</div>
            <p class="card-description">独创的 Session-State 缓存机制，多数据集切换实现快速响应，拒绝冗余计算等待。</p>
        </div>
    </div>

    
    <div class="data-overview-section">
        <div class="section-header">
            <div class="section-title">交互组件</div>
        </div>
        <div class="overview-cards">
            <div class="overview-card">
                <div class="overview-header">
                    <div class="blink-dot"></div>
                    <div class="overview-card-title">空间组分图谱</div>
                </div>
                <div class="overview-module-type">核心模块</div>
            </div>
            <div class="overview-card">
                <div class="overview-header">
                    <div class="blink-dot"></div>
                    <div class="overview-card-title">优势亚群分布</div>
                </div>
                <div class="overview-module-type">实时交互</div>
            </div>
            <div class="overview-card">
                <div class="overview-header">
                    <div class="blink-dot"></div>
                    <div class="overview-card-title">细胞比例概览</div>
                </div>
                <div class="overview-module-type">统计洞察</div>
            </div>
            <div class="overview-card">
                <div class="overview-header">
                    <div class="blink-dot"></div>
                    <div class="overview-card-title">单细胞热度图</div>
                </div>
                <div class="overview-module-type">微观热点</div>
            </div>
            <div class="overview-card">
                <div class="overview-header">
                    <div class="blink-dot"></div>
                    <div class="overview-card-title">原始数据详单</div>
                </div>
                <div class="overview-module-type">导出支持</div>
            </div>
        </div>
    </div>

    <div class="tech-stack-section">
        <div class="tech-stack-header">
            <div class="tech-stack-title">技术栈支持</div>
        </div>
        <div class="marquee-container">
            <div class="marquee-track">
                {tech_items_html}
            </div>
        </div>
    </div>

    <div class="step-guide">
        <div class="step-title">快速启动分析</div>
        <p class="step-desc">点击左上角展开侧边栏，从"选择数据集"中加载现有项目，或通过"📁 导入"上传您的研究数据。</p>
    </div>
</div>
"""
