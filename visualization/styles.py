
import streamlit as st
import os
import base64

def _get_stardust_b64():
    # Try local asset first
    path = os.path.join(os.path.dirname(__file__), "assets", "stardust.png")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"
    # Fallback if file missing
    return "https://www.transparenttextures.com/patterns/stardust.png"

def get_css():
    """
    汇集应用自定义样式表 - 升级至 "Future Lab" 视觉语言 (2025 Edition)。
    包含动态极光背景、全域玻璃拟态、全息悬停效果及影院级动效。
    """
    css_template = """
    <style>
        /* --- 0. 全局字体与重置 --- */
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

        /* 沉浸式布局：顶部保留适度留白 */
        .stMainBlockContainer {
            padding-top: 3rem !important;   /* 恢复顶部留白 */
            padding-bottom: 2rem !important;
            margin-top: 0 !important;       /* 取消上提，让内容自然排列 */
            max-width: 100% !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        /* 隐藏顶部彩虹装饰线 */
        div[data-testid="stDecoration"] {
            display: none;
        }

        /* 恢复 Header 但设为透明 */
        header {
            background: transparent !important;
            backdrop-filter: none !important;
        }
        
        header[data-testid="stHeader"] {
            background: transparent !important;
            border-bottom: none !important;
            box-shadow: none !important;
            height: 3rem !important; /* 减小高度占位 */
            pointer-events: none !important;
        }
        
        /* 恢复 Header 内按钮交互 */
        header[data-testid="stHeader"] button, 
        [data-testid="stSidebarCollapsedControl"],
        .stDeployButton {
            pointer-events: auto !important;
            color: rgba(255, 255, 255, 0.5) !important;
        }
        
        /* --- 1. 沉浸式动态背景 (Aurora Effect) --- */
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
        
        /* 叠加细腻的噪点纹理，增加质感 */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: url("__STARDUST_IMAGE__");
            opacity: 0.4;
            pointer-events: none;
            z-index: 0;
        }

        /* --- 2. 侧边栏：高级磨砂玻璃 --- */
        section[data-testid="stSidebar"] > div:first-child {
            background: rgba(18, 18, 28, 0.65) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
            box-shadow: 10px 0 30px rgba(0, 0, 0, 0.3) !important;
        }
        
        /* 侧边栏标题 - 霓虹流光 */
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

        /* --- 3. 按钮体系：未来主义风格 --- */
        /* Primary Button: 镭射渐变 */
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

        /* Secondary Button: 幽灵边框 + 发光文字 */
        .stButton > button[kind="secondary"],
        .stButton > button[data-testid="baseButton-secondary"],
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: rgba(255, 255, 255, 0.85) !important;
            border-radius: 8px !important;
            backdrop-filter: blur(4px);
            transition: all 0.3s ease !important;
        }
        
        .stButton > button[kind="secondary"]:hover,
        .stButton > button[data-testid="baseButton-secondary"]:hover,
        section[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
            border-color: #667eea !important;
            background: rgba(102, 126, 234, 0.1) !important;
            color: #fff !important;
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.3) !important;
            text-shadow: 0 0 8px rgba(102, 126, 234, 0.6) !important;
        }

        /* --- 4. 仪表盘组件 (Cards & Layout) --- */
        /* 指标卡片 (Metrics) */
        div[data-testid="stMetric"] {
            background: rgba(30, 30, 40, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px !important;
            padding: 15px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
            transition: transform 0.2s ease !important;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02) !important;
            border-color: rgba(102, 126, 234, 0.4) !important;
            background: rgba(40, 40, 60, 0.6) !important;
        }
        
        div[data-testid="stMetricLabel"] > div {
            font-family: 'JetBrains Mono', monospace !important;
            color: rgba(255, 255, 255, 0.5) !important;
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div[data-testid="stMetricValue"] > div {
            color: #fff !important;
            font-weight: 700 !important;
            text-shadow: 0 0 10px rgba(0, 242, 96, 0.3);
        }

        /* --- 5. 落地页炫酷效果 (Holographic Cards) --- */
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
            height: 480px; /* 增加高度，营造电影级宽屏感 */
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
        
        /* 移除只需 hover 的旧样式，现在是自动播放 */
        .banner-container:hover .banner-image {
            /* 保持动画运行，不做额外处理 */
        }
        
        /* 电影级暗角 + 扫描线纹理 */
        .banner-container::after {
            content: '';
            position: absolute;
            inset: 0;
            /* 移除两侧强烈的黑边遮罩，仅在底部保留轻微渐变以衬托文字 */
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

        .hero-section {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -40%); /* 略微上移视觉中心 */
            z-index: 10;
            text-align: center;
            width: 100%;
            padding: 0 2rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        /* 英文装饰线：极简工业风 */
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
            
            /* 冰川质感渐变 */
            background: linear-gradient(180deg, #ffffff 10%, #a5b4fc 60%, #667eea 90%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            
            filter: drop-shadow(0 10px 30px rgba(0, 0, 0, 0.8));
            position: relative;
        }
        
        /* 标题两侧的光标装饰 */
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

        .bio-card:hover::before {
            opacity: 1;
        }

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
        
        .step-guide {
            margin-top: 5rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 20px;
            padding: 3rem;
            border: 1px dashed rgba(255, 255, 255, 0.1);
            text-align: center;
        }

        .step-title {
            color: #00f260;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1rem;
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
    </style>
    """
    return css_template.replace("__STARDUST_IMAGE__", _get_stardust_b64())


def inject_custom_css():
    """
    将封装的 CSS 样式表注入当前 Streamlit 会话上下文。
    """
    st.markdown(get_css(), unsafe_allow_html=True)

def get_landing_page_html(banner_src):
    """
    构造系统初始化引导页面的 HTML 结构。
    """
    return f"""
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
    <div class="features-container">
        <div class="bio-card">
            <i class="fa-solid fa-bolt card-icon-large"></i>
            <div class="card-title-main">WebGL 2.0 加速</div>
            <p class="card-description">底层采用 GPU 加速渲染引擎，支持数万级空间位点实时交互，缩放平移顺滑无阻。</p>
        </div>
        <div class="bio-card">
            <i class="fa-solid fa-microscope card-icon-large"></i>
            <div class="card-title-main">超分辨率反卷积</div>
            <p class="card-description">集成先进的 GNN 架构，提供亚细胞级的组分还原，精准锁定每一个空间位点的细胞构成。</p>
        </div>
        <div class="bio-card">
            <i class="fa-solid fa-database card-icon-large"></i>
            <div class="card-title-main">智能显存缓存</div>
            <p class="card-description">独创的 Session-State 缓存机制，多数据集切换实现毫秒级响应，拒绝冗余计算等待。</p>
        </div>
    </div>
    <div class="step-guide">
        <div class="step-title">快速启动分析</div>
        <p class="step-desc">点击左上角展开侧边栏，从“选择数据集”中加载现有项目，或通过“✨ 导入”上传您的研究数据。</p>
    </div>
</div>
"""
