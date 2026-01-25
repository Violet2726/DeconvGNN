
import streamlit as st

def get_css():
    """
    Get the custom CSS styles for the application.
    """
    return """
    <style>
        /* 隐藏 Streamlit 默认菜单和水印，保留侧边栏按钮 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        .main-header {
            font-size: 1.5rem;
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
    """
    Inject custom CSS into the Streamlit app.
    """
    st.markdown(get_css(), unsafe_allow_html=True)
