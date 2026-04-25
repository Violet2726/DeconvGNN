# -*- coding: utf-8 -*-
"""
STdGCN 核心数据处理工具库。

本模块包含模型训练前的主要数据准备逻辑：表达矩阵预处理、标记基因筛选、
伪斑点模拟，以及真实空间斑点与伪斑点的特征整合。函数尽量保持纯数据
变换职责，便于在训练脚本、可视化脚本和单元验证中复用。
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import multiprocessing
from tqdm.auto import tqdm
import random
from sklearn.decomposition import NMF


from .autoencoder import *



def ST_preprocess(ST_exp, 
                  normalize=True,
                  log=True,
                  highly_variable_genes=False, 
                  regress_out=False, 
                  scale=False,
                  scale_max_value=None,
                  scale_zero_center=True,
                  hvg_min_mean=0.0125,
                  hvg_max_mean=3,
                  hvg_min_disp=0.5,
                  highly_variable_gene_num=None
                 ):
    """
    对空间转录组或单细胞数据进行标准的预处理流程：
    标准化 -> 对数变换 -> 高变基因筛选 -> 回归协变量 -> 缩放。

    参数:
        ST_exp: 输入 AnnData，可来自真实空间斑点或单细胞参考数据。
        normalize: 是否把每个观测的总计数归一化到 1e4。
        log: 是否执行 `log1p` 对数变换。
        highly_variable_genes: 是否筛选高变基因。
        regress_out: 是否回归线粒体比例和总计数等协变量。
        scale: 是否按基因缩放到近似零均值、单位方差。
        scale_max_value: Scanpy 缩放上限。
        scale_zero_center: 是否进行零中心化。
        hvg_min_mean: 高变基因筛选的最小平均表达。
        hvg_max_mean: 高变基因筛选的最大平均表达。
        hvg_min_disp: 高变基因筛选的最小离散度。
        highly_variable_gene_num: 指定保留的高变基因数量。

    返回:
        AnnData: 预处理后的副本，不会修改原始输入对象。
    """
    
    adata = ST_exp.copy()
    
    # Scanpy 的预处理函数会原地修改 AnnData，因此先 copy，避免污染上游数据。
    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        
    if log == True:
        sc.pp.log1p(adata)
        
    # 标记基因筛选默认读取该层，保留对数化后的矩阵作为统计检验输入。
    adata.layers['scale.data'] = adata.X.copy()
    
    if highly_variable_genes == True:
        sc.pp.highly_variable_genes(adata, 
                                    min_mean=hvg_min_mean, 
                                    max_mean=hvg_max_mean, 
                                    min_disp=hvg_min_disp,
                                    n_top_genes=highly_variable_gene_num,
                                   )
        adata = adata[:, adata.var.highly_variable]
        
    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])
    
    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)
    
    return adata



def find_marker_genes(sc_exp, 
                      preprocess = True,
                      highly_variable_genes = True,
                      regress_out = False,
                      scale = False,
                      PCA_components = 50, 
                      marker_gene_method = 'wilcoxon',
                      filter_wilcoxon_marker_genes = True, 
                      top_gene_per_type = 20, 
                      pvals_adj_threshold = 0.10,
                      log_fold_change_threshold = 1,
                      min_within_group_fraction_threshold = 0.7,
                      max_between_group_fraction_threshold = 0.3,
                     ):
    """
    从单细胞参考数据中筛选各细胞类型的标记基因 (Marker Genes)。
    支持 Wilcoxon 秩和检验和 Logistic Regression。

    参数:
        sc_exp: 带有 `obs['cell_type']` 的单细胞 AnnData。
        preprocess: 是否先执行标准预处理。
        highly_variable_genes: 预处理阶段是否筛选高变基因。
        regress_out: 预处理阶段是否回归协变量。
        scale: 预处理阶段是否缩放表达矩阵。
        PCA_components: PCA 组件数量，供 Scanpy 排序流程使用。
        marker_gene_method: `wilcoxon` 或 `logreg`。
        filter_wilcoxon_marker_genes: Wilcoxon 结果是否额外按统计阈值过滤。
        top_gene_per_type: 每个细胞类型最多保留的标记基因数量。
        pvals_adj_threshold: 校正后 p 值阈值。
        log_fold_change_threshold: log fold change 阈值。
        min_within_group_fraction_threshold: 组内表达比例下限。
        max_between_group_fraction_threshold: 组外表达比例上限。

    返回:
        tuple: `(gene_list, gene_dict)`，分别为去重后的基因列表和按细胞类型
        分组的标记基因字典。
    """

    if preprocess == True:
        sc_adata_marker_gene = ST_preprocess(sc_exp.copy(), 
                                             normalize=True,
                                             log=True,
                                             highly_variable_genes=highly_variable_genes, 
                                             regress_out=regress_out, 
                                             scale=scale,
                                            )
    else:
        sc_adata_marker_gene = sc_exp.copy()

    sc.tl.pca(sc_adata_marker_gene, n_comps=PCA_components, svd_solver='arpack', random_state=None)
    
    layer = 'scale.data'
    sc.tl.rank_genes_groups(sc_adata_marker_gene, 'cell_type', layer=layer, use_raw=False, pts=True, 
                            method=marker_gene_method, corr_method='benjamini-hochberg', key_added=marker_gene_method)

    if marker_gene_method == 'wilcoxon':
        if filter_wilcoxon_marker_genes == True:
            gene_dict = {}
            gene_list = []
            for name in sc_adata_marker_gene.obs['cell_type'].unique():
                # Wilcoxon 分支按 p 值、fold change 和表达比例逐层过滤，
                # 以尽量选择“组内高表达、组外低表达”的可靠 marker。
                data = sc.get.rank_genes_groups_df(sc_adata_marker_gene, group=name, key=marker_gene_method).sort_values('pvals_adj')
                if pvals_adj_threshold != None:
                    data = data[data['pvals_adj'] < pvals_adj_threshold]
                if log_fold_change_threshold != None:
                    data = data[data['logfoldchanges'] >= log_fold_change_threshold]
                if min_within_group_fraction_threshold != None:
                    data = data[data['pct_nz_group'] >= min_within_group_fraction_threshold]
                if max_between_group_fraction_threshold != None:
                    data = data[data['pct_nz_reference'] < max_between_group_fraction_threshold]
                gene_dict[name] = data['names'].values[:top_gene_per_type].tolist()
                gene_list = gene_list + data['names'].values[:top_gene_per_type].tolist()
                gene_list = list(set(gene_list))
        else:
            gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
            gene_dict = {}
            for i in gene_table.columns:
                gene_dict[i] = gene_table[i].values.tolist()
            gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    elif marker_gene_method == 'logreg':
        gene_table = pd.DataFrame(sc_adata_marker_gene.uns[marker_gene_method]['names'][:top_gene_per_type])
        gene_dict = {}
        for i in gene_table.columns:
            gene_dict[i] = gene_table[i].values.tolist()
        gene_list = list(set([item   for sublist in gene_table.values.tolist()   for item in sublist]))
    else:
        print("参数 marker_gene_method 必须为 'logreg' 或 'wilcoxon'。")
    
    return gene_list, gene_dict



def generate_a_spot(sc_exp, 
                    min_cell_number_in_spot, 
                    max_cell_number_in_spot,
                    max_cell_types_in_spot,
                    generation_method,
                   ):
    """
    生成单个模拟伪斑点 (Pseudo-spot)。
    从单细胞数据中随机抽取细胞进行聚合。

    参数:
        sc_exp: 单细胞 AnnData，需包含 `cell_type` 和 `cell_type_idx`。
        min_cell_number_in_spot: 每个伪斑点包含的最少细胞数。
        max_cell_number_in_spot: 每个伪斑点包含的最多细胞数。
        max_cell_types_in_spot: 使用 `celltype` 策略时允许的最多细胞类型数。
        generation_method: `cell` 表示按细胞均匀抽样，`celltype` 表示先抽细胞类型再抽细胞。

    返回:
        AnnData: 单个伪斑点对应的细胞子集。
    """
    
    if generation_method == 'cell':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = random.randint(min_cell_number_in_spot, max_cell_number_in_spot)
        cell_type_list = list(sc_exp.obs['cell_type'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)
        
        # 先抽取互不重复的细胞类型，再从这些类型中生成每个细胞的类型标签。
        # 这样可以控制伪斑点的细胞类型复杂度，模拟真实斑点中的混合组成。
        while(True):
            cell_type_list_selected = random.choices(sc_exp.obs['cell_type'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['cell_type'].isin(cell_type_list_selected)]
        
        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['cell_type'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])
            
        return sc_exp_filter[picked_cells]
    else:
        print('参数 generation_method 必须为 "cell" 或 "celltype"。')

        

# def pseudo_spot_generation(sc_exp,
#                            idx_to_word_celltype,
#                            spot_num,
#                            min_cell_number_in_spot,
#                            max_cell_number_in_spot,
#                            max_cell_types_in_spot,
#                            generation_method,
#                            n_jobs = -1
#                           ):
#
#     cell_type_num = len(sc_exp.obs['cell_type'].unique())
#
#     cores = multiprocessing.cpu_count()
#     if n_jobs == -1:
#         pool = multiprocessing.Pool(processes=cores)
#     else:
#         pool = multiprocessing.Pool(processes=n_jobs)
#     args = [(sc_exp, min_cell_number_in_spot, max_cell_number_in_spot, max_cell_types_in_spot, generation_method) for i in range(spot_num)]
#     generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))
#
#     pseudo_spots = []
#     pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
#     pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
#     for i in range(spot_num):
#         one_spot = generated_spots[i]
#         pseudo_spots.append(one_spot)
#         pseudo_spots_table[i] = one_spot.X.sum(axis=0)
#         for j in one_spot.obs.index:
#             type_idx = one_spot.obs.loc[j, 'cell_type_idx']
#             pseudo_fraction_table[i, type_idx] += 1
#     pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
#     pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:,:].values)
#     pseudo_spots.obs.index = pseudo_spots_table.index[:]
#     pseudo_spots.var.index = pseudo_spots_table.columns[:]
#     type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
#     pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
#     pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
#     for i in pseudo_fraction_table.columns[:-1]:
#         pseudo_fraction_table[i] = pseudo_fraction_table[i]/pseudo_fraction_table['cell_num']
#     pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)
#
#     return pseudo_spots


def pseudo_spot_generation(sc_exp,
                           idx_to_word_celltype,
                           spot_num,
                           min_cell_number_in_spot,
                           max_cell_number_in_spot,
                           max_cell_types_in_spot,
                           generation_method,
                           n_jobs=-1  # 兼容历史接口；当前实现使用串行生成以避免多进程序列化成本。
                           ):
    """
    批量生成伪斑点数据集，用于训练模型。

    参数:
        sc_exp: 单细胞参考 AnnData。
        idx_to_word_celltype: 细胞类型整数索引到名称的映射。
        spot_num: 需要生成的伪斑点数量。
        min_cell_number_in_spot: 单个伪斑点的最小细胞数。
        max_cell_number_in_spot: 单个伪斑点的最大细胞数。
        max_cell_types_in_spot: 单个伪斑点最多包含的细胞类型数。
        generation_method: 伪斑点抽样策略。
        n_jobs: 历史兼容参数，当前不启用并行。

    返回:
        AnnData: 伪斑点表达矩阵，obs 中包含各细胞类型比例和 `cell_num`。
    """
    cell_type_num = len(sc_exp.obs['cell_type'].unique())

    # 直接使用循环替代线程池
    generated_spots = []
    for i in tqdm(range(spot_num), desc='正在生成模拟伪斑点 (Generating pseudo-spots)'):
        spot = generate_a_spot(
            sc_exp=sc_exp,
            min_cell_number_in_spot=min_cell_number_in_spot,
            max_cell_number_in_spot=max_cell_number_in_spot,
            max_cell_types_in_spot=max_cell_types_in_spot,
            generation_method=generation_method
        )
        generated_spots.append(spot)

    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)

    # 将每个伪斑点中的细胞表达求和，作为空间斑点级表达；
    # 同时按 cell_type_idx 统计组成比例，作为监督学习标签。
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            pseudo_fraction_table[i, type_idx] += 1

    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = anndata.AnnData(X=pseudo_spots_table.iloc[:, :].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i] / pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)

    return pseudo_spots



def data_integration(real, 
                     pseudo, 
                     batch_removal_method="combat",
                     dimensionality_reduction_method='PCA', 
                     dim=50, 
                     scale=True,
                     autoencoder_epoches=2000,
                     autoencoder_LR=1e-3,
                     autoencoder_drop=0,
                     cpu_num=-1,
                     AE_device='GPU'
                    ):
    """
    整合真实空间数据和伪斑点数据，进行去批次效应 (Batch Correction) 和降维。
    支持方法: MNN, Combat, Scanorama。

    参数:
        real: 真实空间斑点 AnnData。
        pseudo: 伪斑点 AnnData。
        batch_removal_method: 批次校正方法，支持 `mnn`、`scanorama`、`combat` 或 `None`。
        dimensionality_reduction_method: 降维方法，支持 `PCA`、`autoencoder`、`nmf` 或 `None`。
        dim: 输出特征维度。
        scale: 是否对整合后的表达矩阵或嵌入做缩放。
        autoencoder_epoches: 自编码器训练轮数。
        autoencoder_LR: 自编码器学习率。
        autoencoder_drop: 自编码器 Dropout 概率。
        cpu_num: CPU 线程数。
        AE_device: 自编码器训练设备。

    返回:
        DataFrame: 前三列固定为 `ST_type`、`cell_num`、`cell_type_num`，
        后续列为整合后的特征。
    """
    
    if batch_removal_method == 'mnn':
        # MNN 会先按互最近邻校正批次，再根据配置进行降维。
        mnn = sc.external.pp.mnn_correct(pseudo, real, svd_dim=dim, k=50, batch_key='real_pseudo', save_raw=True, var_subset=None)
        adata = mnn[0]
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table = table.iloc[pseudo.shape[0]:,:].append(table.iloc[:pseudo.shape[0],:])
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'scanorama':
        # Scanorama 直接在 AnnData 的 obsm 中写入集成嵌入。
        import scanorama
        scanorama.integrate_scanpy([real, pseudo], dimred = dim)
        table1 = pd.DataFrame(real.obsm['X_scanorama'], index=real.obs.index.values)
        table2 = pd.DataFrame(pseudo.obsm['X_scanorama'], index=pseudo.obs.index.values)
        table = pd.concat([table1, table2])
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    elif batch_removal_method == 'combat':
        # Combat 需要先合并两个域，并通过 batch_key 标记真实/伪斑点来源。
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        sc.pp.combat(adata, key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    else:
        # 不做批次校正时仅拼接两个域；该路径常用于快速实验或已同源的数据集。
        aaa = real.copy()
        aaa.obs = pd.DataFrame(index = aaa.obs.index)
        bbb = pseudo.copy()
        bbb.obs = pd.DataFrame(index = bbb.obs.index)
        adata = aaa.concatenate(bbb, batch_key='real_pseudo')
        if dimensionality_reduction_method == 'PCA':
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=True)
            sc.tl.pca(adata, n_comps=dim, svd_solver='arpack', random_state=None)
            table = pd.DataFrame(adata.obsm['X_pca'], index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'autoencoder':
            data = torch.tensor(adata.X)
            x_size = data.shape[1]
            latent_size = dim
            hidden_size = int((x_size + latent_size)/2)
            nets = autoencoder(x_size=x_size, hidden_size=hidden_size, embedding_size=latent_size, p_drop=autoencoder_drop)
            optimizer_ae = torch.optim.Adam(nets.parameters(), lr=autoencoder_LR)
            loss_ae = nn.MSELoss(reduction = 'mean')
            embedding = auto_train(model=nets, 
                                   epoch_n=autoencoder_epoches, 
                                   loss_fn=loss_ae, 
                                   optimizer=optimizer_ae, 
                                   data=data,
                                   cpu_num=cpu_num,
                                   device=AE_device
                                  ).detach().numpy()
            if scale == True:
                embedding = (embedding-embedding.mean(axis=0))/embedding.std(axis=0)
            table = pd.DataFrame(embedding, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == 'nmf':
            nmf = NMF(n_components=dim).fit_transform(adata.X)
            if scale == True:
                nmf = (nmf-nmf.mean(axis=0))/nmf.std(axis=0)
            table = pd.DataFrame(nmf, index=[str(i)[:-2] for i in adata.obs.index])
        elif dimensionality_reduction_method == None:
            if scale == True:
                sc.pp.scale(adata, max_value=None, zero_center=False)
            table = pd.DataFrame(adata.X, index=[str(i)[:-2] for i in adata.obs.index], columns=adata.var.index.values)
        table.insert(0, 'ST_type', ['real']*real.shape[0]+['pseudo']*pseudo.shape[0])
        
    # 保留细胞数量和细胞类型数量作为元数据列，后续特征矩阵会从第 4 列开始读取。
    table.insert(1, 'cell_num', real.obs['cell_num'].values.tolist()+pseudo.obs['cell_num'].values.tolist())
    table.insert(2, 'cell_type_num', real.obs['cell_type_num'].values.tolist()+pseudo.obs['cell_type_num'].values.tolist())

    return table
