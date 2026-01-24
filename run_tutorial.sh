#!/bin/bash
echo "[INFO] =========================================="
echo "[INFO] 启动 STdGCN 演示 (Linux)"
echo "[INFO] =========================================="

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "[ERROR] 找不到 python 命令，请确保已安装 Python 并激活了环境。"
    exit 1
fi

echo "[INFO] 检查环境配置..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

echo ""
echo "[INFO] 正在运行 Tutorial.py ..."
echo "=================================================="
python Tutorial.py
echo "=================================================="

echo ""
echo "[INFO] 运行结束。"
