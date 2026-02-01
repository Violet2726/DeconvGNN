#!/usr/bin/env python
# coding: utf-8


import os
import sys
import warnings
import numpy as np
import torch

import random
import time
import secrets
import argparse

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from core.STdGCN import run_STdGCN
from visualization.utils import handle_visualization_generation
import pandas as pd

secure_random = secrets.randbelow(2 ** 32)

# 2. 使用高精度时间 (纳秒级)
high_prec_time = int(time.time() * 1e9) % 2 ** 32

# 3. 加入进程ID
process_id = os.getpid() % 2 ** 16

# 4. 使用Python内置随机状态的初始熵
random_state = hash(str(random.getstate()[1][:3])) % 2 ** 32

# 组合所有源 (使用异或操作增强随机性)
seed = secure_random ^ high_prec_time ^ (process_id << 16) ^ random_state

# 确保在合法范围内
seed = seed % (2 ** 32)

seed = 7931225
# 设置Python内置随机模块
random.seed(seed)

# 设置NumPy随机种子
np.random.seed(seed)

# 设置PyTorch随机种子
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 多GPU时使用

# 设置cuDNN配置 (可能会降低性能)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 设置Python哈希种子 (如果使用哈希的场合)
os.environ['PYTHONHASHSEED'] = str(seed)

'''
该模块用于提供加载数据和保存数据的路径。

参数:
sc_path: 加载单细胞参考数据的路径。
ST_path: 加载空间转录组数据的路径。
output_path: 保存输出文件的路径。

加载所需的相关文件名和数据格式:
sc_data.tsv: 单细胞参考数据的表达矩阵，行为细胞，列为基因。该文件应保存在"sc_path"中。
sc_label.tsv: 单细胞数据的细胞类型注释。该表格应包含两列：细胞条形码/名称和细胞类型注释信息。
            该文件应保存在"sc_path"中。
ST_data.tsv: 空间转录组数据的表达矩阵，行为斑点，列为基因。该文件应保存在"ST_path"中。
coordinates.csv: 空间转录组数据的坐标。该表格应包含三列：斑点条形码/名称、X轴（列名'x'）和Y轴（列名'y'）。
            该文件应保存在"ST_path"中。
marker_genes.tsv [可选]: 用于运行STdGCN的基因列表。每行一个基因，不允许有表头。该文件应保存在"sc_path"中。
ST_ground_truth.tsv [可选]: ST数据的真实标签。数据应转换为细胞类型比例。该文件应保存在"ST_path"中。
'''
# 解析命令行参数
parser = argparse.ArgumentParser(description='STdGCN 训练/推理脚本')
parser.add_argument('--dataset', type=str, default='CytAssist_11mm_FFPE_Mouse_Embryo', 
                    help='要运行的数据集名称 (例如: V1_Mouse_Brain_Sagittal_Posterior)')
parser.add_argument('--generate_plot', type=bool, default=True, 
                    help='训练完成后是否生成用于Web可视化的Top-4饼图背景')
args = parser.parse_args()

# 数据集名称
dataset_name = args.dataset

paths = {
    'sc_path': f'./data/{dataset_name}/combined', 
    'ST_path': f'./data/{dataset_name}/combined',
    'output_path': f'./data/{dataset_name}/results',
}

'''
该模块用于预处理输入数据并识别标记基因 [可选]。

参数:
'preprocess': [bool]. 选择是否需要预处理输入的表达数据。此步骤包括归一化、对数化、选择高变基因、
                    回归掉线粒体基因和缩放数据。
'normalize': [bool]. 当'preprocess'=True时，选择是否需要将每个细胞/斑点的总计数归一化为10,000，
                    以便归一化后每个细胞/斑点具有相同的总计数。
'log': [bool]. 当'preprocess'=True时，选择是否需要对表达矩阵进行对数化处理（X=log(X+1)）。
'highly_variable_genes': [bool]. 当'preprocess'=True时，选择是否需要筛选高变基因。
'highly_variable_gene_num': [int或None]. 当'preprocess'=True且'highly_variable_genes'=True时，
                    选择要保留的高变基因数量。
'regress_out': [bool]. 当'preprocess'=True时，选择是否需要回归掉线粒体基因。
'scale': [bool]. 当'preprocess'=True时，选择是否需要将每个基因缩放至单位方差和零均值。
'PCA_components': [int]. 主成分分析（PCA）计算的主成分数量。
'marker_gene_method': ['logreg', 'wilcoxon']. 我们使用"scanpy.tl.rank_genes_groups"
                    (https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.rank_genes_groups.html)
                    来识别细胞类型标记基因。对于标记基因选择，STdGCN提供两种方法：
                    'wilcoxon'（Wilcoxon秩和检验）和'logreg'（使用逻辑回归）。
'top_gene_per_type': [int]. 每种细胞类型可用于训练STdGCN的基因数量。
'filter_wilcoxon_marker_genes': [bool]. 当'marker_gene_method'='wilcoxon'时，选择是否需要额外的基因筛选步骤。
'pvals_adj_threshold': [float或None]. 当'marker_gene_method'='wilcoxon'且'rank_gene_filter'=True时，
                    仅保留校正p值 < 'pvals_adj_threshold'的基因。
'log_fold_change_threshold': [float或None]. 当'marker_gene_method'='wilcoxon'且'rank_gene_filter'=True时，
                    仅保留对数倍数变化 > 'log_fold_change_threshold'的基因。
'min_within_group_fraction_threshold': [float或None]. 当'marker_gene_method'='wilcoxon'且'rank_gene_filter'=True时，
                    仅保留在细胞类型内表达比例至少为'min_within_group_fraction_threshold'的基因。
'max_between_group_fraction_threshold': [float或None]. 当'marker_gene_method'='wilcoxon'且'rank_gene_filter'=True时，
                    仅保留在其余细胞类型联合表达比例最多为'max_between_group_fraction_threshold'的基因。
'''
find_marker_genes_paras = {
    'preprocess': True,  # 是否对输入数据进行预处理
    'normalize': True,  # 是否归一化数据
    'log': True,  # 是否对数据进行对数转换
    'highly_variable_genes': False,  # 是否选择高变基因
    'highly_variable_gene_num': None,  # 高变基因数量
    'regress_out': False,  # 是否回归掉线粒体基因
    'PCA_components': 30,  # PCA降维的维度数
    'marker_gene_method': 'logreg',  # 标记基因选择方法（logreg或wilcoxon）
    'top_gene_per_type': 50,  # 每种细胞类型选择的标记基因数量（100）
    'filter_wilcoxon_marker_genes': True,  # 是否过滤wilcoxon方法选择的标记基因
    'pvals_adj_threshold': 0.10,  # 校正p值阈值
    'log_fold_change_threshold': 1,  # 对数倍数变化阈值
    'min_within_group_fraction_threshold': None,  # 组内表达比例阈值
    'max_between_group_fraction_threshold': None,  # 组间表达比例阈值
}

'''
该模块用于模拟伪斑点。

参数:
'spot_num': [int]. 伪斑点的数量。
'min_cell_num_in_spot': [int]. 伪斑点中的最小细胞数。
'max_cell_num_in_spot': [int]. 伪斑点中的最大细胞数。
'generation_method': ['cell'或'celltype']. STdGCN提供两种伪斑点模拟方法。
                    当'generation_method'='cell'时，每个细胞被等概率选择。
                    当'generation_method'='celltype'时，每种细胞类型被等概率选择。
                    详见论文了解更多细节。
'max_cell_types_in_spot': [int]. 当'generation_method'='celltype'时，选择伪斑点中细胞类型的最大数量。
'''
pseudo_spot_simulation_paras = {
    'spot_num': 15000,  # 生成的伪斑点数量（30000）
    'min_cell_num_in_spot': 8,  # 每个伪斑点中的最小细胞数
    'max_cell_num_in_spot': 12,  # 每个伪斑点中的最大细胞数
    'generation_method': 'celltype',  # 生成方法（cell或celltype）
    'max_cell_types_in_spot': 4,  # 每个伪斑点中的最大细胞类型数
}

'''
该模块用于真实斑点和伪斑点的归一化。

参数:
'normalize': [bool]. 选择是否需要将每个细胞/斑点的总计数归一化为10,000，
                    以便归一化后每个细胞/斑点具有相同的总计数。
'log': [bool]. 选择是否需要对表达矩阵进行对数化处理（X=log(X+1)）。
'scale': [bool]. 选择是否需要将每个基因缩放至单位方差和零均值。
'''
data_normalization_paras = {
    'normalize': True,  # 是否归一化数据
    'log': True,  # 是否对数转换
    'scale': False,  # 是否缩放数据
}

'''
该模块用于整合归一化后的真实斑点和伪斑点，以构建真实到伪斑点的连接图。

参数:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. 考虑到批次效应，STdGCN提供四种整合方法：
                    mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3),
                    combat (Combat, DOI: 10.1093/biostatistics/kxj037),
                    None（不进行批次移除的拼接）。
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. 当'batch_removal_method'不是'scanorama'时，
                    选择数据是否需要降维以及应用哪种降维方法。
'dim': [int]. 当'batch_removal_method'='scanorama'时，选择该方法的维度。
                    当'batch_removal_method'不是'scanorama'且'dimensionality_reduction_method'不是None时，
                    选择降维的维度。
'scale': [bool]. 当'batch_removal_method'不是'scanorama'时，选择是否需要将每个基因缩放至单位方差和零均值。
'''
integration_for_adj_paras = {
    'batch_removal_method': None,
    'dim': 30,
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}

'''
该模块用于构建表达图的邻接矩阵，包含三个子图：真实到伪斑点的图、伪斑点内部的图和真实斑点内部的图。

参数:
'find_neighbor_method' ['MNN', 'KNN']. STdGCN提供两种连接图构建方法：
                    KNN（K最近邻）和MNN（互最近邻，DOI: 10.1038/nbt.4091）。
'dist_method': ['euclidean', 'cosine']. 用于计算斑点之间配对距离的度量方法。
'corr_dist_neighbors': [int]. 最近邻的数量。
'PCA_dimensionality_reduction': [bool]. 对于伪斑点内部图和真实斑点内部图的构建，
                    选择在计算斑点之间的配对距离之前是否需要使用PCA降维。
'dim': [int]. 当'PCA_dimensionality_reduction'=True时，选择PCA的维度。
'''
inter_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 20,
}
real_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 10,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}
pseudo_intra_exp_adj_paras = {
    'find_neighbor_method': 'MNN',
    'dist_method': 'cosine',
    'corr_dist_neighbors': 20,
    'PCA_dimensionality_reduction': False,
    'dim': 50,
}

'''
该模块用于构建空间图的邻接矩阵。

参数:
'space_dist_threshold': [float或None]. 只有距离小于'space_dist_threshold'的两个斑点才能连接。
'link_method' ['soft', 'hard']. 如果斑点i和j连接，当'link_method'='hard'时，A(i,j)=1；
                    当'link_method'='soft'时，A(i,j)=1/distance(i,j)。详见论文了解更多细节。
'''
spatial_adj_paras = {
    'link_method': 'soft',
    'space_dist_threshold': 2,
}

'''
该模块用于整合归一化后的真实斑点和伪斑点作为STdGCN的输入特征。

参数:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. 考虑到批次效应，STdGCN提供四种整合方法：
                    mnn (mnnpy, DOI:10.1038/nbt.4091),
                    scanorama (Scanorama, DOI: 10.1038/s41587-019-0113-3),
                    combat (Combat, DOI: 10.1093/biostatistics/kxj037),
                    None（不进行批次移除的拼接）。
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. 当'batch_removal_method'不是'scanorama'时，
                    选择数据是否需要降维以及应用哪种降维方法。
'dim': [int]. 当'batch_removal_method'='scanorama'时，选择该方法的维度。
                    当'batch_removal_method'不是'scanorama'且'dimensionality_reduction_method'不是None时，
                    选择降维的维度。
'scale': [bool]. 当'batch_removal_method'不是'scanorama'时，选择是否需要将每个基因缩放至单位方差和零均值。
'''
integration_for_feature_paras = {
    'batch_removal_method': None,
    'dimensionality_reduction_method': None,
    'dim': 80,
    'scale': True,
}

'''
该模块用于设置STdGCN的深度学习参数。

参数:
'epoch_n': [int]. 最大训练轮数。
'dim': [int]. 隐藏层的维度。
'common_hid_layers_num': [int]. GCN层数 = 'common_hid_layers_num'+1。
'fcnn_hid_layers_num': [int]. 全连接神经网络层数 = 'fcnn_hid_layers_num'+2。
'dropout': [float]. 元素被置零的概率。
'learning_rate_SGD': [float]. 初始学习率。
'weight_decay_SGD': [float]. L2惩罚系数。
'momentum': [float]. 动量因子。
'dampening': [float]. 动量阻尼项。
'nesterov': [bool]. 是否启用Nesterov动量。
'early_stopping_patience': [int]. 早停轮数。
'clip_grad_max_norm': [float]. 裁剪参数迭代器的梯度范数。
#'LambdaLR_scheduler_coefficient': [float]. LambdaLR调度器函数的系数:
#                    lr(epoch) = [LambdaLR_scheduler_coefficient] ^ epoch_n × learning_rate_SGD.
'print_loss_epoch_step': [int]. 每'print_epoch_step'轮打印一次损失值。
'''
GCN_paras = {
    'epoch_n': 3000,  # 训练轮数
    'dim': 80,  # 隐藏层维度
    'common_hid_layers_num': 1,  # GCN层数量
    'fcnn_hid_layers_num': 1,  # 全连接层数量
    'dropout': 0,  # dropout比例
    'learning_rate_SGD': 2e-1,  # 学习率
    'weight_decay_SGD': 3e-4,  # L2正则化系数
    'momentum': 0.9,  # 动量
    'dampening': 0,  # 动量阻尼
    'nesterov': True,  # 是否使用Nesterov动量
    'early_stopping_patience': 20,  # 早停耐心值
    'clip_grad_max_norm': 1,  # 梯度裁剪最大范数
    'print_loss_epoch_step': 20,  # 打印损失的轮数间隔
}

'''
## 运行STdGCN

参数
'load_test_groundtruth': [bool]. 选择是否需要上传空间转录组数据的真实标签文件（ST_ground_truth.tsv）
                    以跟踪STdGCN的性能。
'use_marker_genes': [bool]. 选择是否需要在运行STdGCN之前进行基因选择过程。否则使用单细胞和空间转录组数据的共有基因。
'external_genes': [bool]. 当"use_marker_genes"=True时，您可以上传自己指定的基因列表（marker_genes.tsv）来运行STdGCN。
'generate_new_pseudo_spots': [bool]. STdGCN会将模拟的伪斑点保存为"pseudo_ST.pkl"。
                    如果您想使用相同的单细胞参考数据运行多次反卷积，您不需要模拟新的伪斑点，
                    可以将'generate_new_pseudo_spots'设置为False。
                    当'generate_new_pseudo_spots'=False时，您需要将"pseudo_ST.pkl"预先移动到'output_path'，
                    以便STdGCN可以直接加载预模拟的伪斑点。
'fraction_pie_plot': [bool]. 选择是否需要绘制预测结果的饼图。根据我们的经验，当预测斑点数量非常大时，
                    我们不建议绘制饼图。对于1,000个斑点，绘图时间不到2分钟；
                    对于2,000个斑点，绘图时间约为10分钟；对于3,000个斑点，
                    则需要约30分钟。
'cell_type_distribution_plot': [bool]. 选择是否需要为每种细胞类型绘制预测结果的散点图。
'n_jobs': [int]. 设置CPU上用于内联并行的线程数。'n_jobs=-1'表示使用所有CPU。
'GCN_device': ['GPU', 'CPU']. 选择用于运行GCN网络的设备。
'''
results = run_STdGCN(
    paths=paths,
    load_test_groundtruth=False,  # 是否加载测试集真实标签
    use_marker_genes=True,  # 是否使用标记基因
    external_genes=False,  # 是否使用外部提供的基因列表
    find_marker_genes_paras=find_marker_genes_paras,
    generate_new_pseudo_spots=True,  # 新数据集需要生成新的伪斑点
    pseudo_spot_simulation_paras=pseudo_spot_simulation_paras,
    data_normalization_paras=data_normalization_paras,
    integration_for_adj_paras=integration_for_adj_paras,
    inter_exp_adj_paras=inter_exp_adj_paras,
    spatial_adj_paras=spatial_adj_paras,
    real_intra_exp_adj_paras=real_intra_exp_adj_paras,
    pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,
    integration_for_feature_paras=integration_for_feature_paras,
    GCN_paras=GCN_paras,
    fraction_pie_plot=False,  # 是否生成饼图可视化
    cell_type_distribution_plot=False,  # 是否生成细胞类型分布散点图
    n_jobs=-1,  # 使用的CPU线程数
    GCN_device='GPU',  # 使用的设备（GPU或CPU）
)

# 保存结果
results.write_h5ad(os.path.join(paths['output_path'], 'results.h5ad'))

# [Visualization] 生成用于 Web 可视化的资源
if args.generate_plot:
    handle_visualization_generation(paths)
