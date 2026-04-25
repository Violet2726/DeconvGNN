#!/usr/bin/env python
# coding: utf-8
"""
Windows 环境下的 STdGCN 训练示例脚本。

与 `Tutorial.py` 相比，本脚本保留 `freeze_support()`，便于 Windows 在涉及
多进程或打包执行时正常启动。参数含义与主教程一致，默认路径指向示例
`./data/sc_data`、`./data/ST_data` 和 `./output`。
"""

import os
import sys
import warnings
from multiprocessing import freeze_support

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from core.STdGCN import run_STdGCN

'''
路径配置说明。

参数:
sc_path: 单细胞参考数据所在目录。
ST_path: 空间转录组数据所在目录。
output_path: 训练输出文件保存目录。

输入文件约定:
sc_data.tsv: 单细胞参考表达矩阵，行是细胞，列是基因，应保存在 sc_path。
sc_label.tsv: 单细胞类型标签表，至少包含细胞条形码/名称和 cell_type 两列，
            应保存在 sc_path。
ST_data.tsv: 空间转录组表达矩阵，行是空间斑点，列是基因，应保存在 ST_path。
coordinates.csv: 空间斑点坐标表，包含斑点条形码/名称、x 坐标和 y 坐标，
            应保存在 ST_path。
marker_genes.tsv [可选]: 外部标记基因列表，每行一个基因且不包含表头，应保存在 sc_path。
ST_ground_truth.tsv [可选]: 空间数据真实细胞类型比例，用于评估测试损失，应保存在 ST_path。
'''
paths = {
    'sc_path': './data/sc_data',
    'ST_path': './data/ST_data',
    'output_path': './output',
}

'''
标记基因筛选参数。

参数:
'preprocess': [bool]. 是否在筛选标记基因前预处理表达矩阵，流程包括归一化、对数化、
                    高变基因筛选、回归线粒体基因影响和缩放。
'normalize': [bool]. 当 preprocess=True 时，是否把每个细胞/斑点的总计数归一化为 10,000。
'log': [bool]. 当 preprocess=True 时，是否执行 log(X+1) 对数变换。
'highly_variable_genes': [bool]. 当 preprocess=True 时，是否筛选高变基因。
'highly_variable_gene_num': [int 或 None]. 高变基因筛选开启时保留的基因数量。
'regress_out': [bool]. 是否回归线粒体基因等协变量。
'scale': [bool]. 是否将每个基因缩放到零均值、单位方差。
'PCA_components': [int]. PCA 主成分数量。
'marker_gene_method': ['logreg', 'wilcoxon']. 使用 scanpy.tl.rank_genes_groups
                    识别细胞类型标记基因；支持 Wilcoxon 秩和检验和逻辑回归。
'top_gene_per_type': [int]. 每个细胞类型最多用于训练的标记基因数量。
'filter_wilcoxon_marker_genes': [bool]. 使用 wilcoxon 时是否额外按统计阈值过滤。
'pvals_adj_threshold': [float 或 None]. 校正后 p 值阈值。
'log_fold_change_threshold': [float 或 None]. log fold change 阈值。
'min_within_group_fraction_threshold': [float 或 None]. 组内表达比例下限。
'max_between_group_fraction_threshold': [float 或 None]. 其他细胞类型联合表达比例上限。
'''
find_marker_genes_paras = {
    'preprocess': True,
    'normalize': True,
    'log': True,
    'highly_variable_genes': False,
    'highly_variable_gene_num': None,
    'regress_out': False,
    'PCA_components': 30,
    'marker_gene_method': 'logreg',
    'top_gene_per_type': 100,
    'filter_wilcoxon_marker_genes': True,
    'pvals_adj_threshold': 0.10,
    'log_fold_change_threshold': 1,
    'min_within_group_fraction_threshold': None,
    'max_between_group_fraction_threshold': None,
}

'''
伪斑点模拟参数。

参数:
'spot_num': [int]. 需要生成的伪斑点数量。
'min_cell_num_in_spot': [int]. 单个伪斑点包含的最少细胞数。
'max_cell_num_in_spot': [int]. 单个伪斑点包含的最多细胞数。
'generation_method': ['cell' 或 'celltype']. 伪斑点抽样策略：
                    'cell' 表示所有细胞等概率抽样；
                    'celltype' 表示先抽细胞类型，再从对应类型中抽细胞。
'max_cell_types_in_spot': [int]. 使用 'celltype' 策略时，单个伪斑点最多包含的细胞类型数。
'''
pseudo_spot_simulation_paras = {
    'spot_num': 3000,
    'min_cell_num_in_spot': 8,
    'max_cell_num_in_spot': 12,
    'generation_method': 'celltype',
    'max_cell_types_in_spot': 4,
}

'''
真实斑点与伪斑点共用的表达预处理参数。

参数:
'normalize': [bool]. 是否把每个细胞/斑点的总计数归一化为 10,000。
'log': [bool]. 是否执行 log(X+1) 对数变换。
'scale': [bool]. 是否按基因缩放到零均值、单位方差。
'''
data_normalization_paras = {
    'normalize': True,
    'log': True,
    'scale': False,
}

'''
用于构建表达邻接图的数据整合参数。

参数:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. 批次效应校正方法：
                    mnn、scanorama、combat，或 None（直接拼接，不做批次校正）。
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. 非 scanorama 路径下使用的降维方法。
'dim': [int]. 整合后特征维度。
'scale': [bool]. 是否对整合后的表达矩阵或低维嵌入进行缩放。
'''
integration_for_adj_paras = {
    'batch_removal_method': None,
    'dim': 30,
    'dimensionality_reduction_method': 'PCA',
    'scale': True,
}

'''
表达图邻接矩阵参数。

表达图包含三类子图：真实斑点到伪斑点的跨域图、伪斑点内部图、真实斑点内部图。

参数:
'find_neighbor_method' ['MNN', 'KNN']. 近邻建图方法，KNN 为 K 近邻，MNN 为互最近邻。
'dist_method': ['euclidean', 'cosine']. 计算斑点间表达距离的度量。
'corr_dist_neighbors': [int]. 近邻数量。
'PCA_dimensionality_reduction': [bool]. 构建域内表达图前是否先做 PCA 降维。
'dim': [int]. PCA 降维维度。
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
空间图邻接矩阵参数。

参数:
'space_dist_threshold': [float 或 None]. 仅当两个斑点距离小于该阈值时才连边。
'link_method' ['soft', 'hard']. hard 表示边权为 1；soft 表示边权为 1/distance。
'''
spatial_adj_paras = {
    'link_method': 'soft',
    'space_dist_threshold': 2,
}

'''
GNN 输入特征整合参数。

该部分与“建图前整合”相互独立：前者服务邻接关系构建，当前参数服务模型输入特征。

参数:
'batch_removal_method': ['mnn', 'scanorama', 'combat', None]. 批次效应校正方法。
'dimensionality_reduction_method': ['PCA', 'autoencoder', 'nmf', None]. 降维方法。
'dim': [int]. 输出特征维度。
'scale': [bool]. 是否对特征进行缩放。
'''
integration_for_feature_paras = {
    'batch_removal_method': None,
    'dimensionality_reduction_method': None,
    'dim': 80,
    'scale': True,
}

'''
STdGCN 深度学习训练参数。

参数:
'epoch_n': [int]. 最大训练轮数。
'dim': [int]. 隐藏层维度。
'common_hid_layers_num': [int]. 图卷积隐藏层数量，实际 GCN 层数为 common_hid_layers_num + 1。
'fcnn_hid_layers_num': [int]. 全连接隐藏层数量，实际输出头层数为 fcnn_hid_layers_num + 2。
'dropout': [float]. Dropout 概率。
'learning_rate_SGD': [float]. SGD 初始学习率。
'weight_decay_SGD': [float]. L2 正则化系数。
'momentum': [float]. 动量系数。
'dampening': [float]. 动量阻尼。
'nesterov': [bool]. 是否启用 Nesterov 动量。
'early_stopping_patience': [int]. 验证损失连续未改善时的早停耐心轮数。
'clip_grad_max_norm': [float]. 梯度裁剪最大范数。
#'LambdaLR_scheduler_coefficient': [float]. LambdaLR 调度器系数:
#                    lr(epoch) = LambdaLR_scheduler_coefficient ^ epoch_n * learning_rate_SGD。
'print_loss_epoch_step': [int]. 每隔多少轮打印一次损失。
'''
GCN_paras = {
    'epoch_n': 3000,
    'dim': 80,
    'common_hid_layers_num': 1,
    'fcnn_hid_layers_num': 1,
    'dropout': 0,
    'learning_rate_SGD': 2e-1,
    'weight_decay_SGD': 3e-4,
    'momentum': 0.9,
    'dampening': 0,
    'nesterov': True,
    'early_stopping_patience': 20,
    'clip_grad_max_norm': 1,
    'print_loss_epoch_step': 20,
}

'''
运行 STdGCN。

参数:
'load_test_groundtruth': [bool]. 是否读取 ST_ground_truth.tsv 以跟踪测试集损失。
'use_marker_genes': [bool]. 是否在训练前筛选标记基因；否则使用单细胞和空间数据共同基因。
'external_genes': [bool]. use_marker_genes=True 时，是否使用外部 marker_genes.tsv。
'generate_new_pseudo_spots': [bool]. 是否重新生成伪斑点并保存为 pseudo_ST.pkl；
                    若设为 False，需要预先把 pseudo_ST.pkl 放到 output_path。
'fraction_pie_plot': [bool]. 是否生成预测结果饼图。大数据集绘制耗时较长，通常建议使用 Web 可视化资源。
'cell_type_distribution_plot': [bool]. 是否为每个细胞类型绘制预测分布散点图。
'n_jobs': [int]. CPU 并行线程数，-1 表示使用全部 CPU。
'GCN_device': ['GPU', 'CPU']. GCN 训练设备。
'''
if __name__ == '__main__':
    freeze_support()
    results = run_STdGCN(paths,
                         load_test_groundtruth=False,
                         use_marker_genes=True,
                         external_genes=False,
                         find_marker_genes_paras=find_marker_genes_paras,
                         generate_new_pseudo_spots=True,
                         pseudo_spot_simulation_paras=pseudo_spot_simulation_paras,
                         data_normalization_paras=data_normalization_paras,
                         integration_for_adj_paras=integration_for_adj_paras,
                         inter_exp_adj_paras=inter_exp_adj_paras,
                         spatial_adj_paras=spatial_adj_paras,
                         real_intra_exp_adj_paras=real_intra_exp_adj_paras,
                         pseudo_intra_exp_adj_paras=pseudo_intra_exp_adj_paras,
                         integration_for_feature_paras=integration_for_feature_paras,
                         GCN_paras=GCN_paras,
                         fraction_pie_plot=True,
                         cell_type_distribution_plot=True,
                         n_jobs=1,
                         GCN_device='GPU'
                         )

    results.write_h5ad(paths['output_path'] + '/results.h5ad')
