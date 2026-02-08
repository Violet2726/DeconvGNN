# **DeconvGNN：基于图神经网络的空间转录组反卷积系统**

> 基于图神经网络（GNN）的空间转录组学反卷积方法与可视化分析系统。以单细胞 RNA 测序数据为参考，解析空间点位中不同细胞类型的组成比例。

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deconvgnn-vis.streamlit.app/)

## 🌐 在线演示

无需本地环境即可体验：
**[https://deconvgnn-vis.streamlit.app/](https://deconvgnn-vis.streamlit.app/)**

> 💡 **提示**：在线版支持直接上传分析结果（需包含 `predict_result.csv` 和 `coordinates.csv`）。

## 🌟 项目亮点

- **交互式可视化平台**：基于 Streamlit，支持缩放、悬停与多层饼图背景呈现。
- **智能配色策略**：基于细胞类型空间相关性的层次聚类，分配相近色相。
- **性能优化体系**：PatchCollection 批量绘制 + WebGL 渲染，适配大样本数据。
- **批量处理支持**：工具脚本支持多数据集批处理流程。
- **全中文体验**：源码注解、日志提示与界面文案均中文化。

---

## 🛠️ 环境要求

推荐 Python 3.8+，核心依赖如下：

```
torch >= 1.11.0
scanpy >= 1.9.1
pandas >= 1.3.5
numpy >= 1.21.6
plotly
streamlit
matplotlib
scipy
tqdm
scikit-learn
```

---

## 🚀 完整流程

### 📥 Step 1：下载数据

```bash
python utils/download_visium_data.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

支持数据集：
- `V1_Adult_Mouse_Brain_Coronal_Section_1`
- `V1_Mouse_Brain_Sagittal_Anterior`
- `V1_Mouse_Brain_Sagittal_Posterior`
- `CytAssist_11mm_FFPE_Mouse_Embryo`

批量下载：
```bash
python utils/download_visium_data.py
```

---

### 🔗 Step 2：整合数据

```bash
python utils/prepare_combined_data.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

批量处理：
```bash
python utils/prepare_combined_data.py
```

---

### 🧠 Step 3：模型训练

```bash
python Tutorial.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

训练完成后会自动生成可视化背景图。

---

### 🎨 Step 4：启动可视化

```bash
python -m streamlit run visualization/app.py
```

也可直接运行：`run_visualization.bat`

---

### 🔄 Step 5：重新生成图表（可选）

```bash
python utils/generate_plot.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

批量处理：
```bash
python utils/generate_plot.py
```

---

## 📁 项目结构

```
DeconvGNN/
├── data/                          # 数据目录
│   ├── ref_mouse_cortex_allen/    # 单细胞参考数据
│   └── [数据集名]/
│       ├── ST_data.tsv            # 空间表达矩阵
│       ├── coordinates.csv        # 空间坐标
│       ├── combined/              # 整合后的训练数据
│       └── results/               # 训练结果与可视化资源
├── core/                          # 核心算法模块
│   ├── STdGCN.py                  # 主算法入口
│   ├── GCN.py                     # 图神经网络模型
│   ├── CKGC.py                    # CKGConv 图卷积层
│   └── adjacency_matrix.py        # 邻接矩阵构建
├── visualization/                 # 可视化模块
│   ├── app.py                     # Streamlit 主程序
│   ├── data_loader.py             # 数据加载与校验
│   ├── viz_utils.py               # 可视化绘图工具
│   └── styles.py                  # 界面样式
├── utils/                         # 工具脚本
│   ├── download_visium_data.py    # 数据下载
│   ├── prepare_combined_data.py   # 数据整合
│   ├── generate_plot.py           # 图表生成
│   └── update_labels.py           # 标签更新
└── Tutorial.py                    # 训练入口
```

---

## 📦 输出产物

结果保存在 `data/[数据集名]/results/`：

| 文件名 | 说明 |
| :--- | :--- |
| `predict_result.csv` | 每个空间点的细胞类型比例预测结果 |
| `Loss_function.jpg` | 训练损失曲线图 |
| `model_parameters` | PyTorch 模型权重 |
| `interactive_pie_background.png` | Web 端交互式饼图底图 |

---

## 📖 参考引用

Li Y, Luo Y. Stdgcn: spatial transcriptomic cell-type deconvolution using graph convolutional networks. *Genome Biol.* (2024) 25:206. [DOI: 10.1186/s13059-024-03353-0](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03353-0)
