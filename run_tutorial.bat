@echo off
chcp 65001 > nul
setlocal

echo [INFO] ==========================================
echo [INFO] 启动 STdGCN 演示
echo [INFO] ==========================================

REM 设置本地 Python 路径
set "PYTHON_EXE=%~dp0_miniconda\python.exe"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] 找不到 Python 环境: %PYTHON_EXE%
    echo 请确认 _miniconda 文件夹是否完整安装。
    pause
    exit /b 1
)

echo [INFO] 检查环境配置...
"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo.
echo [INFO] 正在运行 Tutorial.py ...
echo ==================================================
"%PYTHON_EXE%" Tutorial.py
echo ==================================================

echo.
echo [INFO] 运行结束。请检查上方是否有报错。
pause
