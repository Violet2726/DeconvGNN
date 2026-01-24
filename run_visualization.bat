@echo off
chcp 65001 >nul
echo ==========================================
echo   STdGCN 可视化系统启动脚本
echo ==========================================
echo.

cd /d "%~dp0"

echo [INFO] 检查 Streamlit 是否已安装...
"%~dp0_miniconda\python.exe" -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [INFO] 正在安装 Streamlit...
    "%~dp0_miniconda\python.exe" -m pip install streamlit plotly -q
)

echo [INFO] 启动可视化系统...
echo.
echo =========================================
echo   浏览器将自动打开，如未打开请访问:
echo   http://localhost:8501
echo =========================================
echo.

"%~dp0_miniconda\python.exe" -m streamlit run visualization_app/app.py --server.port 8501

pause
