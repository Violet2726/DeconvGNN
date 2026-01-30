<img src="https://github.com/luoyuanlab/stdgcn/blob/main/img_folder/Logo.jpg" height="160px" />  <br />

# **iSTdGCN-Vis: 空间转录组反卷积改进版可视化系统**

> 本项目基于 **STdGCN (Genome Biology, 2024)** 框架进行了深度的功能增强与可视化重构，旨在提供更直观、更高效的空间分布分析体验。

## 🌟 核心改进
*   **交互式 Web 可视化平台**：基于 Streamlit 构建，支持空间跨尺度缩放、多层级饼图背景展现及实时悬停比例查看。
*   **可视化逻辑高度集成**：将原有的分散绘图代码重构为统一的 `visualization` 模块，显著提升维护性。
*   **极致性能优化**：
    *   使用 `PatchCollection` 优化多色饼图渲染，大幅减少由于大样本量导致的绘图卡顿。
    *   Plotly 散点图悬停算法优化，数千个点的实时交互无延迟。
*   **极简操作流程**：内置一键启动脚本（`.bat`），支持训练完成后自动生成可视化资源。
*   **全面中文化支持**：源代码注释、控制台输出及 Web 界面文字均已重构为中文。

---

## 🛠️ 环境配置
推荐使用 Python 3.8+ 环境，安装以下核心依赖：
- `torch == 1.11.0`
- `scanpy == 1.9.1`
- `pandas == 1.3.5`
- `numpy == 1.21.6`
- `plotly`
- `streamlit`
- `matplotlib`
- `scipy`
- `tqdm`
- `sklearn`

---

## 🚀 快速入门

### 1. 模型训练 (Training)
运行 `Tutorial.py` 进行模型训练。它会自动读取 `data` 目录下的输入数据，训练完成后会保存预测结果并触发可视化背景图生成。
```bash
python Tutorial.py --dataset CytAssist_11mm_FFPE_Mouse_Embryo
```
或者在 Windows 环境下直接双击运行：`run_tutorial.bat`

### 2. 启动可视化系统 (Visualization)
训练完成后，运行以下命令启动交互式 Web 界面，从浏览器查看分析结果：
```bash
python -m streamlit run visualization/app.py
```
或者在 Windows 环境下直接双击运行：`run_visualization.bat`

---

## 📁 数据输入规范
系统要求的数据格式位于每个数据集的 `combined/` 目录下：
*   **sc_data.tsv**: 单细胞参考表达矩阵（细胞 × 基因）。
*   **sc_label.tsv**: 对应的细胞类型注释文件（Barcode 与 Cell_Type）。
*   **ST_data.tsv**: 空间转录组表达矩阵（Spot × 基因）。
*   **coordinates.csv**: 空间坐标文件，要求包含 `x`, `y` 两列。

---

## 📦 输出产物
结果默认保存在 `data/[数据集名]/results/`：
*   `predict_result.csv`: 最终预测的每个位置的细胞类型比例。
*   `Loss_function.jpg`: 训练损失值下降曲线图。
*   `model_parameters`: 训练好的 PyTorch 模型权重。
*   `interactive_pie_background.png`: 专为 Web 端交互优化的多色饼图底图。

---

## 📖 参考引用
[1] Li Y, Luo Y. Stdgcn: spatial transcriptomic cell-type deconvolution using graph convolutional networks. *Genome Biol.* (2024) 25:206. [DOI: 10.1186/s13059-024-03353-0](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03353-0)
