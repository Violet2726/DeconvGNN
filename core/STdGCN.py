# -*- coding: utf-8 -*-
"""
STdGCN 训练主流程。

该模块只保留一条面向外部调用的编排入口：`run_STdGCN`。它负责把
单细胞参考数据、空间转录组数据、伪斑点模拟、邻接矩阵构建、特征整合
以及 GNN 训练串联起来。具体算法实现分散在 `utils.py`、
`adjacency_matrix.py` 和 `GCN.py` 中，本文件主要承担流程协调职责。
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import pickle

from .utils import *
from .autoencoder import *
from .adjacency_matrix import *
from .GCN import *


   

def run_STdGCN(paths,
               find_marker_genes_paras,
               pseudo_spot_simulation_paras,
               data_normalization_paras,
               integration_for_adj_paras,
               inter_exp_adj_paras,
               spatial_adj_paras,
               real_intra_exp_adj_paras,
               pseudo_intra_exp_adj_paras,
               integration_for_feature_paras,
               GCN_paras,
               load_test_groundtruth = False,
               use_marker_genes = True,
               external_genes = False,
               generate_new_pseudo_spots = True,
               fraction_pie_plot = True,
               cell_type_distribution_plot = True,
               n_jobs = -1,
               GCN_device = 'CPU'
              ):
    """
    运行完整的 STdGCN 反卷积流程。

    参数:
        paths: 数据路径字典，需包含 `sc_path`、`ST_path` 和 `output_path`。
        find_marker_genes_paras: 标记基因筛选参数。
        pseudo_spot_simulation_paras: 伪斑点模拟参数。
        data_normalization_paras: 真实斑点与伪斑点共用的预处理参数。
        integration_for_adj_paras: 构建表达图前的数据整合参数。
        inter_exp_adj_paras: 真实斑点与伪斑点之间表达邻接图的参数。
        spatial_adj_paras: 空间距离邻接图参数。
        real_intra_exp_adj_paras: 真实斑点域内表达邻接图参数。
        pseudo_intra_exp_adj_paras: 伪斑点域内表达邻接图参数。
        integration_for_feature_paras: 生成 GNN 输入特征时的数据整合参数。
        GCN_paras: GNN 结构与训练参数。
        load_test_groundtruth: 是否读取空间斑点真实比例，用于测试集损失监控。
        use_marker_genes: 是否先筛选标记基因再训练。
        external_genes: 是否直接使用 `marker_genes.tsv` 中的外部基因列表。
        generate_new_pseudo_spots: 是否重新模拟伪斑点；关闭时复用输出目录中的缓存。
        fraction_pie_plot: 保留的兼容参数，当前主流程不直接绘图。
        cell_type_distribution_plot: 保留的兼容参数，当前主流程不直接绘图。
        n_jobs: Scanpy、PyTorch 等步骤可使用的 CPU 线程数，`-1` 表示自动。
        GCN_device: GNN/自编码器训练设备，通常为 `CPU` 或 `GPU`。

    返回:
        AnnData: 真实空间斑点的 AnnData，其中 `obsm['predict_result']`
        保存模型输出的细胞类型比例。
    """

    sc_path = paths['sc_path']
    ST_path = paths['ST_path']
    output_path = paths['output_path']
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # 1. 读取单细胞参考数据，并将文本标签映射为连续整数索引。
    #    这些整数索引后续用于累计伪斑点中的细胞类型比例。
    sc_adata = sc.read_csv(sc_path+"/sc_data.tsv", delimiter='\t')
    sc_label = pd.read_table(sc_path+"/sc_label.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
    sc_label.columns = ['cell_type']
    sc_adata.obs['cell_type'] = sc_label['cell_type'].values

    cell_type_num = len(sc_adata.obs['cell_type'].unique())
    cell_types = sc_adata.obs['cell_type'].unique()

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}
    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    celltype_idx = [word_to_idx_celltype[w] for w in sc_adata.obs['cell_type']]
    sc_adata.obs['cell_type_idx'] = celltype_idx
    sc_adata.obs['cell_type'].value_counts()

    # 2. 确定参与训练的基因集合。
    #    使用标记基因可以降低噪声和训练维度；若传入外部基因列表，则保持用户指定顺序。
    if use_marker_genes == True:
        if external_genes == True:
            with open(sc_path+"/marker_genes.tsv", 'r') as f:
                selected_genes = [line.rstrip('\n') for line in f]
        else:
            selected_genes, cell_type_marker_genes = find_marker_genes(sc_adata,
                                                                      preprocess = find_marker_genes_paras['preprocess'],
                                                                      highly_variable_genes = find_marker_genes_paras['highly_variable_genes'],
                                                                      PCA_components = find_marker_genes_paras['PCA_components'], 
                                                                      filter_wilcoxon_marker_genes = find_marker_genes_paras['filter_wilcoxon_marker_genes'], 
                                                                      marker_gene_method = find_marker_genes_paras['marker_gene_method'],
                                                                      pvals_adj_threshold = find_marker_genes_paras['pvals_adj_threshold'],
                                                                      log_fold_change_threshold = find_marker_genes_paras['log_fold_change_threshold'],
                                                                      min_within_group_fraction_threshold = find_marker_genes_paras['min_within_group_fraction_threshold'],
                                                                      max_between_group_fraction_threshold = find_marker_genes_paras['max_between_group_fraction_threshold'],
                                                                      top_gene_per_type = find_marker_genes_paras['top_gene_per_type'])
            with open(output_path+"/marker_genes.tsv", 'w') as f:
                for gene in selected_genes:
                    f.write(str(gene) + '\n')
            
    print("已选择 {} 个基因作为标记基因。".format(len(selected_genes)))
    
    

    # 3. 伪斑点是监督信号的来源：表达矩阵由抽样细胞聚合而来，
    #    obs 中的细胞类型比例则作为 GNN 的训练标签。
    if generate_new_pseudo_spots == True:
        pseudo_adata = pseudo_spot_generation(sc_adata,
                                              idx_to_word_celltype,
                                              spot_num = pseudo_spot_simulation_paras['spot_num'],
                                              min_cell_number_in_spot = pseudo_spot_simulation_paras['min_cell_num_in_spot'],
                                              max_cell_number_in_spot = pseudo_spot_simulation_paras['max_cell_num_in_spot'],
                                              max_cell_types_in_spot = pseudo_spot_simulation_paras['max_cell_types_in_spot'],
                                              generation_method = pseudo_spot_simulation_paras['generation_method'],
                                              n_jobs = n_jobs
                                              )
        data_file = open(output_path+'/pseudo_ST.pkl','wb')
        pickle.dump(pseudo_adata, data_file)
        data_file.close()
    else:
        data_file = open(output_path+'/pseudo_ST.pkl','rb')
        pseudo_adata = pickle.load(data_file)
        data_file.close()

    # 4. 读取真实空间表达矩阵和坐标。坐标只附加到真实斑点，
    #    不参与伪斑点监督标签构造。
    ST_adata = sc.read_csv(ST_path+"/ST_data.tsv", delimiter='\t')
    ST_coor = pd.read_table(ST_path+"/coordinates.csv", sep = ',', header = 0, index_col = 0, encoding = "utf-8")
    ST_coor.index = ST_coor.index.astype(str)
    ST_adata.obs['coor_X'] = ST_coor['x']
    ST_adata.obs['coor_Y'] = ST_coor['y']
    if load_test_groundtruth == True:
        ST_groundtruth = pd.read_table(ST_path+"/ST_ground_truth.tsv", sep = '\t', header = 0, index_col = 0, encoding = "utf-8")
        for i in cell_types:
            ST_adata.obs[i] = ST_groundtruth[i]

    # 5. 真实空间数据与伪斑点只保留共同基因，保证后续矩阵维度一致。
    ST_genes = ST_adata.var.index.values
    pseudo_genes = pseudo_adata.var.index.values
    common_genes = set(ST_genes).intersection(set(pseudo_genes))
    ST_adata_filter = ST_adata[:,list(common_genes)]
    pseudo_adata_filter = pseudo_adata[:,list(common_genes)]
    
    
    # 6. 分别预处理真实斑点和伪斑点；二者必须使用同一套归一化策略，
    #    否则构建表达图和训练标签时会引入系统性偏差。
    ST_adata_filter_norm = ST_preprocess(ST_adata_filter, 
                                         normalize = data_normalization_paras['normalize'], 
                                         log = data_normalization_paras['log'], 
                                         scale = data_normalization_paras['scale'],
                                        )

    genes_to_keep = ST_adata_filter_norm.var_names[ST_adata_filter_norm.var_names.isin(selected_genes)]
    ST_adata_filter_norm = ST_adata_filter_norm[:, genes_to_keep]
    
    # 真实空间数据通常没有细胞数和比例标签；这里补齐占位字段，
    # 使真实斑点与伪斑点的 obs schema 对齐，便于后续拼接标签矩阵。
    try:
        try:
            ST_adata_filter_norm.obs.insert(0, 'cell_num', ST_adata_filter.obs['cell_num'])
        except:
            ST_adata_filter_norm.obs['cell_num'] = ST_adata_filter.obs['cell_num']
    except:
        ST_adata_filter_norm.obs.insert(0, 'cell_num', [0]*ST_adata_filter_norm.obs.shape[0])
    for i in cell_types:
        try:
            ST_adata_filter_norm.obs[i] = ST_adata_filter.obs[i]
        except:
            ST_adata_filter_norm.obs[i] = [0]*ST_adata_filter_norm.obs.shape[0]
    try:
        ST_adata_filter_norm.obs['cell_type_num'] = (ST_adata_filter_norm.obs[cell_types]>0).sum(axis=1)
    except:
        ST_adata_filter_norm.obs['cell_type_num'] = [0]*ST_adata_filter_norm.obs.shape[0]


    pseudo_adata_norm = ST_preprocess(pseudo_adata_filter, 
                                      normalize = data_normalization_paras['normalize'], 
                                      log = data_normalization_paras['log'], 
                                      scale = data_normalization_paras['scale'],
                                     )

    pseudo_adata_norm = pseudo_adata_norm[:, genes_to_keep]

    pseudo_adata_norm.obs['cell_type_num'] = (pseudo_adata_norm.obs[cell_types]>0).sum(axis=1)
    
    
    # 7. 为表达邻接图构建整合空间。此处的降维/批次校正只服务于“建图”，
    #    与稍后作为 GNN 输入的特征整合相互独立。
    ST_integration = data_integration(ST_adata_filter_norm, 
                                      pseudo_adata_norm, 
                                      batch_removal_method = integration_for_adj_paras['batch_removal_method'], 
                                      dim = min(integration_for_adj_paras['dim'], int(ST_adata_filter_norm.shape[1]/2)), 
                                      dimensionality_reduction_method=integration_for_adj_paras['dimensionality_reduction_method'],
                                      scale=integration_for_adj_paras['scale'],
                                      cpu_num=n_jobs,
                                      AE_device=GCN_device
                                      )
    
    A_inter_exp =  inter_adj(ST_integration, 
                             find_neighbor_method=inter_exp_adj_paras['find_neighbor_method'], 
                             dist_method=inter_exp_adj_paras['dist_method'], 
                             corr_dist_neighbors=inter_exp_adj_paras['corr_dist_neighbors'], 
                            )

    A_intra_space = intra_dist_adj(ST_adata_filter_norm, 
                                   link_method=spatial_adj_paras['link_method'],
                                   space_dist_threshold=spatial_adj_paras['space_dist_threshold'],
                                  )
    
    A_real_intra_exp = intra_exp_adj(ST_adata_filter_norm, 
                                     find_neighbor_method=real_intra_exp_adj_paras['find_neighbor_method'], 
                                     dist_method=real_intra_exp_adj_paras['dist_method'],
                                     PCA_dimensionality_reduction=real_intra_exp_adj_paras['PCA_dimensionality_reduction'],
                                     corr_dist_neighbors=real_intra_exp_adj_paras['corr_dist_neighbors'],
                                    )

    A_pseudo_intra_exp = intra_exp_adj(pseudo_adata_norm, 
                                       find_neighbor_method=pseudo_intra_exp_adj_paras['find_neighbor_method'], 
                                       dist_method=pseudo_intra_exp_adj_paras['dist_method'],
                                       PCA_dimensionality_reduction=pseudo_intra_exp_adj_paras['PCA_dimensionality_reduction'],
                                       corr_dist_neighbors=pseudo_intra_exp_adj_paras['corr_dist_neighbors'],
                                      )
    
    # 8. 将三类邻接关系放入统一的大图：
    #    - inter_exp: 真实斑点与伪斑点之间的表达相似连接；
    #    - pseudo/real intra_exp: 两个域内部的表达相似连接；
    #    - intra_space: 真实斑点内部的空间邻近连接。
    real_num = ST_adata_filter.shape[0]
    pseudo_num = pseudo_adata.shape[0]

    adj_inter_exp = A_inter_exp.values
    adj_pseudo_intra_exp = A_intra_transfer(A_pseudo_intra_exp, 'pseudo', real_num, pseudo_num)
    adj_real_intra_exp = A_intra_transfer(A_real_intra_exp, 'real', real_num, pseudo_num)
    adj_intra_space = A_intra_transfer(A_intra_space, 'real', real_num, pseudo_num)

    # 表达图与空间图使用两个通道分别输入模型。
    # 对角增强项用于保留节点自身信息，diag_power 用于控制邻接边相对自环的强度。
    adj_alpha = 1
    adj_beta = 1
    diag_power = 20
    adj_balance = (1+adj_alpha+adj_beta)*diag_power
    adj_exp = torch.tensor(adj_inter_exp+adj_alpha*adj_pseudo_intra_exp+adj_beta*adj_real_intra_exp)/adj_balance + torch.eye(adj_inter_exp.shape[0])
    adj_sp = torch.tensor(adj_intra_space)/diag_power + torch.eye(adj_intra_space.shape[0])

    # norm = True
    # if(norm == True):
    #     adj_exp = torch.tensor(adj_normalize(adj_exp, symmetry=True))
    #     adj_sp = torch.tensor(adj_normalize(adj_sp, symmetry=True))
        
        
    # 9. 生成 GNN 节点特征。与建图不同，这里输出的矩阵会直接输入神经网络，
    #    因此维度会受模型隐藏层大小约束。
    ST_integration_batch_removed = data_integration(ST_adata_filter_norm, 
                                                    pseudo_adata_norm, 
                                                    batch_removal_method=integration_for_feature_paras['batch_removal_method'], 
                                                    dim=min(int(ST_adata_filter_norm.shape[1]*1/2), integration_for_feature_paras['dim']), 
                                                    dimensionality_reduction_method=integration_for_feature_paras['dimensionality_reduction_method'], 
                                                    scale=integration_for_feature_paras['scale'],
                                                    cpu_num=n_jobs,
                                                    AE_device=GCN_device
                                                   )
    feature = torch.tensor(ST_integration_batch_removed.iloc[:, 3:].values)
    
    
    # 10. 初始化模型、优化器和学习率调度器。
    input_layer = feature.shape[1]
    hidden_layer = min(int(ST_adata_filter_norm.shape[1]*1/2), GCN_paras['dim'])
    output_layer1 = len(word_to_idx_celltype)
    epoch_n = GCN_paras['epoch_n']
    common_hid_layers_num = GCN_paras['common_hid_layers_num']
    fcnn_hid_layers_num = GCN_paras['fcnn_hid_layers_num']
    dropout = GCN_paras['dropout']
    learning_rate_SGD = GCN_paras['learning_rate_SGD']
    weight_decay_SGD = GCN_paras['weight_decay_SGD']
    momentum = GCN_paras['momentum']
    dampening = GCN_paras['dampening']
    nesterov = GCN_paras['nesterov']
    early_stopping_patience = GCN_paras['early_stopping_patience']
    clip_grad_max_norm = GCN_paras['clip_grad_max_norm']
    LambdaLR_scheduler_coefficient = 0.997
    ReduceLROnPlateau_factor = 0.1
    ReduceLROnPlateau_patience = 5
    scheduler = 'scheduler_ReduceLROnPlateau'
    print_epoch_step = GCN_paras['print_loss_epoch_step']
    cpu_num = n_jobs
    
    model = conGCN(nfeat = input_layer, 
                   nhid = hidden_layer, 
                   common_hid_layers_num = common_hid_layers_num, 
                   fcnn_hid_layers_num = fcnn_hid_layers_num, 
                   dropout = dropout, 
                   nout1 = output_layer1
                  )

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr = learning_rate_SGD, 
                                momentum = momentum, 
                                weight_decay = weight_decay_SGD, 
                                dampening = dampening, 
                                nesterov = nesterov)
    
    scheduler_LambdaLR = torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                           lr_lambda = lambda epoch: LambdaLR_scheduler_coefficient ** epoch)
    scheduler_ReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                             mode='min', 
                                                                             factor=ReduceLROnPlateau_factor, 
                                                                             patience=ReduceLROnPlateau_patience, 
                                                                             threshold=0.0001, 
                                                                             threshold_mode='rel', 
                                                                             cooldown=0, 
                                                                             min_lr=0)
    if scheduler == 'scheduler_LambdaLR':
        scheduler = scheduler_LambdaLR
    elif scheduler == 'scheduler_ReduceLROnPlateau':
        scheduler = scheduler_ReduceLROnPlateau
    else:
        scheduler = None
    
    loss_fn1 = nn.KLDivLoss(reduction = 'mean')

    # 11. 标签矩阵的顺序必须与特征矩阵一致：
    #     前半部分是真实斑点（测试/推理），后半部分是伪斑点（训练/验证）。
    train_valid_len = pseudo_adata.shape[0]
    test_len = ST_adata_filter.shape[0]

    table1 = ST_adata_filter_norm.obs.copy()
    label1 = pd.concat([table1[pseudo_adata.obs.iloc[:,:-1].columns], pseudo_adata.obs.iloc[:,:-1]])
    label1 = torch.tensor(label1.values)

    adjs = [adj_exp.float(), adj_sp.float()]

    # 12. 只在伪斑点上计算训练/验证损失，真实斑点作为待预测对象；
    #     若提供 ground truth，则额外记录测试损失但不参与反向传播。
    output1, loss, trained_model = conGCN_train(model = model, 
                                                train_valid_len = train_valid_len,
                                                train_valid_ratio = 0.9,
                                                test_len = test_len, 
                                                feature = feature, 
                                                adjs = adjs, 
                                                label = label1, 
                                                epoch_n = epoch_n, 
                                                loss_fn = loss_fn1, 
                                                optimizer = optimizer, 
                                                scheduler = scheduler, 
                                                early_stopping_patience = early_stopping_patience,
                                                clip_grad_max_norm = clip_grad_max_norm,
                                                load_test_groundtruth = load_test_groundtruth,
                                                print_epoch_step = print_epoch_step,
                                                cpu_num = cpu_num,
                                                GCN_device = GCN_device
                                               )
    
    # 13. 保存训练曲线、预测比例表、模型参数和 AnnData 结果，
    #     这些文件同时服务后续分析和 Streamlit 可视化。
    loss_table = pd.DataFrame(loss, columns=['train', 'valid', 'test'])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(loss_table.index, loss_table['train'], label='训练集')
    ax.plot(loss_table.index, loss_table['valid'], label='验证集')
    if load_test_groundtruth == True:
        ax.plot(loss_table.index, loss_table['test'], label='测试集')
    ax.set_xlabel('训练轮数 (Epoch)', fontsize = 20)
    ax.set_ylabel('损失值 (Loss)', fontsize = 20)
    ax.set_title('损失函数曲线', fontsize = 20)
    ax.legend(fontsize = 15)
    plt.tight_layout()
    plt.savefig(output_path+'/Loss_function.jpg', dpi=300)
    plt.close('all')
    
    predict_table = pd.DataFrame(np.exp(output1[:test_len].detach().numpy()).tolist(), index=ST_adata_filter_norm.obs.index, columns=pseudo_adata_norm.obs.columns[:-2])
    predict_table.to_csv(output_path+'/predict_result.csv', index=True, header=True)
    
    torch.save(trained_model, output_path+'/model_parameters')
    
    pred_use = np.round(output1.exp().detach()[:test_len].cpu().numpy(), decimals=4)
    cell_type_list = cell_types
    coordinates = ST_adata_filter_norm.obs[['coor_X', 'coor_Y']]
    
    ST_adata_filter_norm.obsm['predict_result'] = np.exp(output1[:test_len].detach().numpy())
    
    torch.cuda.empty_cache()
    
    return ST_adata_filter_norm
