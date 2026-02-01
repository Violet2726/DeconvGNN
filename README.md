# **DeconvGNN: 基于图神经网络的空间转录组反卷积系统**

> 一种基于图神经网络（GNN）的空间转录组学数据反卷积方法及可视化分析系统。以单细胞 RNA 测序数据作为参考，解析空间转录组学点位中不同细胞类型的组成比例。

## 🌟 核心改进

*   **交互式 Web 可视化平台**：基于 Streamlit 构建，支持空间跨尺度缩放、多层级饼图背景展现及实时悬停比例查看。
*   **智能配色系统**：基于细胞类型空间相关性的层次聚类，自动分配相似颜色给空间分布相近的细胞类型。
*   **极致性能优化**：
    *   使用 `PatchCollection` 优化多色饼图渲染，大幅减少由于大样本量导致的绘图卡顿。
    *   Plotly 散点图悬停算法优化，数千个点的实时交互无延迟。
*   **批量处理支持**：所有工具脚本均支持一次性处理多个数据集。
*   **全面中文化支持**：源代码注释、控制台输出及 Web 界面文字均已重构为中文。

---

## 🛠️ 环境配置

推荐使用 Python 3.8+ 环境，安装以下核心依赖：

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

## 🚀 完整工作流程

### 📥 Step 1: 下载数据

下载 10x Visium 官方示例数据集：

```bash
python utils/download_visium_data.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

**支持的数据集：**
- `V1_Adult_Mouse_Brain_Coronal_Section_1`
- `V1_Mouse_Brain_Sagittal_Anterior`
- `V1_Mouse_Brain_Sagittal_Posterior`
- `CytAssist_11mm_FFPE_Mouse_Embryo`

**批量下载：**
```bash
python utils/download_visium_data.py  # 不加参数则下载全部
```

---

### 🔗 Step 2: 整合数据

将单细胞参考数据与空间数据整合，生成训练所需的 `combined/` 目录：

```bash
python utils/prepare_combined_data.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

**批量处理：**
```bash
python utils/prepare_combined_data.py  # 不加参数则处理全部
```

---

### 🧠 Step 3: 模型训练

运行 STdGCN 模型进行细胞类型反卷积：

```bash
python Tutorial.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

训练完成后会自动生成可视化背景图。

---

### 🎨 Step 4: 启动可视化

启动交互式 Web 界面查看分析结果：

```bash
python -m streamlit run visualization/app.py
```

或者直接双击运行：`run_visualization.bat`

---

### 🔄 Step 5: 重新生成图表（可选）

如需在不重新训练的情况下更新可视化：

```bash
python utils/generate_plot.py --dataset V1_Adult_Mouse_Brain_Coronal_Section_1
```

**批量处理：**
```bash
python utils/generate_plot.py  # 不加参数则处理全部
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
│   └── utils.py                   # 绑图工具函数
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
