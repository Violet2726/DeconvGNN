# -*- coding: utf-8 -*-
"""
STdGCN 邻接矩阵构建模块。

该模块负责把预处理后的表达特征和空间坐标转换为图结构。STdGCN 使用
两类图：表达相似图用于连接真实斑点与伪斑点，空间邻近图用于刻画真实
空间斑点的物理邻接关系。所有函数均返回与输入顺序对齐的方阵，便于后续
拼接成统一大图。
"""
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import KDTree
from sklearn.metrics import DistanceMetric
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors



def find_mutual_nn(data1, 
                   data2, 
                   dist_method, 
                   k1, 
                   k2, 
                  ):
    """
    计算两组数据之间的互最近邻（Mutual Nearest Neighbors, MNN）。

    参数:
        data1: 第一组样本矩阵。
        data2: 第二组样本矩阵。
        dist_method: 距离度量名称；`cosine` 会走相似度分支，其余使用 KDTree。
        k1: 对 data1 查询 data2 时保留的邻居数。
        k2: 对 data2 查询 data1 时保留的邻居数。

    返回:
        list: 每个元素为 `[data1_index, data2_index]` 的互最近邻配对。
    """
    if dist_method == 'cosine':
        cos_sim1 = cosine_similarity(data1, data2)
        cos_sim2 = cosine_similarity(data2, data1)
        k_index_1 = torch.topk(torch.tensor(cos_sim2.astype("float")), k=k2, dim=1)[1]
        k_index_2 = torch.topk(torch.tensor(cos_sim1.astype("float")), k=k1, dim=1)[1]
    else:
        dist = DistanceMetric.get_metric(dist_method)
        k_index_1 = KDTree(data1, metric=dist).query(data2, k=k2, return_distance=False)
        k_index_2 = KDTree(data2, metric=dist).query(data1, k=k1, return_distance=False)
    # 只有当两个方向都把对方视为近邻时才认为配对可靠，
    # 这可以降低批次差异或噪声导致的单向误连接。
    mutual_1 = []
    mutual_2 = []
    mutual = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]: 
                mutual_1.append(index_1)
                mutual_2.append(index_2)
                mutual.append([index_1, index_2])
    return mutual



def inter_adj(ST_integration, 
              find_neighbor_method='MNN',
              dist_method='euclidean',
              corr_dist_neighbors=20, 
             ):
    """
    构建跨域邻接矩阵，即真实空间斑点与伪斑点之间的连接。

    参数:
        ST_integration: `data_integration` 输出表，需包含 `ST_type` 和降维特征列。
        find_neighbor_method: `KNN` 或 `MNN`。
        dist_method: 表达空间中的距离度量。
        corr_dist_neighbors: 每个节点参与近邻搜索的邻居数。

    返回:
        DataFrame: 真实斑点与伪斑点共同节点空间中的对称邻接矩阵。
    """
    
    if find_neighbor_method == 'KNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        # KNN 分支：每个真实斑点连接到表达空间中最接近的若干伪斑点。
        if dist_method == 'cosine':
            cos_sim = cosine_similarity(data1, data2)
            k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
        else:
            dist = DistanceMetric.get_metric(dist_method)
            k_index = KDTree(data2, metric=dist).query(data1, k=corr_dist_neighbors, return_distance=False)
        A_exp = np.zeros((ST_integration.shape[0], ST_integration.shape[0]), dtype=float)
        for i in range(k_index.shape[0]):
            for j in k_index[i]:
                A_exp[i, j+real_num] = 1;
                A_exp[j+real_num, i] = 1;  
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)
        
    elif find_neighbor_method == 'MNN':
        real = ST_integration[ST_integration['ST_type'] == 'real']
        pseudo = ST_integration[ST_integration['ST_type'] == 'pseudo']
        data1 = real.iloc[:, 3:]
        data2 = pseudo.iloc[:, 3:]
        # MNN 分支更保守，要求真实斑点和伪斑点互相出现在近邻集合中。
        mut = find_mutual_nn(data2, data1, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
        mut = pd.DataFrame(mut, columns=['pseudo', 'real'])
        real_num = real.shape[0]
        pseudo_num = pseudo.shape[0]
        A_exp = np.zeros((real_num+pseudo_num, real_num+pseudo_num), dtype=float)
        for i in mut.index:
            A_exp[mut.loc[i, 'real'], mut.loc[i, 'pseudo']+real_num] = 1
            A_exp[mut.loc[i, 'pseudo']+real_num, mut.loc[i, 'real']] = 1
        A_exp = pd.DataFrame(A_exp, index=ST_integration.index, columns=ST_integration.index)
    
    return A_exp



def intra_dist_adj(ST_exp, 
                   link_method='soft',
                   space_dist_neighbors=27, 
                   space_dist_threshold=None
                  ):
    """
    基于空间坐标构建真实斑点的域内空间邻接矩阵。

    参数:
        ST_exp: 带有 `obs['coor_X']` 和 `obs['coor_Y']` 的 AnnData。
        link_method: `hard` 表示二值连接，`soft` 表示距离倒数加权。
        space_dist_neighbors: 每个斑点参与空间近邻搜索的邻居数。
        space_dist_threshold: 可选距离阈值，超过阈值的近邻会被忽略。

    返回:
        DataFrame: 与真实斑点顺序对齐的空间邻接矩阵。
    """
    
    knn = NearestNeighbors(n_neighbors=space_dist_neighbors, metric='minkowski')

    knn.fit(ST_exp.obs[['coor_X', 'coor_Y']])
    dist, ind = knn.kneighbors()
    
    if link_method == 'hard':
        # hard 连接只记录是否相邻，适合不希望距离大小影响消息强度的实验。
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i,j] < space_dist_threshold:
                        A_space[i, ind[i,j]] = 1
                        A_space[ind[i,j], i] = 1
                else:
                    A_space[i, ind[i,j]] = 1
                    A_space[ind[i,j], i] = 1
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        # soft 连接用距离倒数表示边权，使近距离斑点在空间图中贡献更大。
        A_space = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                if space_dist_threshold != None:
                    if dist[i,j] < space_dist_threshold:
                        A_space[i, ind[i,j]] = 1 / dist[i,j]
                        A_space[ind[i,j], i] = 1 / dist[i,j]
                else:
                    A_space[i, ind[i,j]] = 1 / dist[i,j]
                    A_space[ind[i,j], i] = 1 / dist[i,j]
        A_space = pd.DataFrame(A_space, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    
    return A_space



def intra_exp_adj(adata, 
                  find_neighbor_method='KNN', 
                  dist_method='euclidean', 
                  PCA_dimensionality_reduction=True, 
                  dim=50, 
                  corr_dist_neighbors=10, 
                  ):
    """
    基于基因表达构建单一域内部的表达邻接矩阵。

    参数:
        adata: 真实斑点或伪斑点 AnnData。
        find_neighbor_method: `KNN` 或 `MNN`。
        dist_method: 表达空间中的距离度量。
        PCA_dimensionality_reduction: 是否先做 PCA 再计算距离。
        dim: PCA 维度。
        corr_dist_neighbors: 近邻数量。

    返回:
        DataFrame: 与输入 AnnData 观测顺序一致的表达邻接矩阵。
    """
        
    ST_exp = adata.copy()
    
    # 建图前统一缩放表达量，避免高表达基因主导距离计算。
    sc.pp.scale(ST_exp, max_value=None, zero_center=True)
    if PCA_dimensionality_reduction == True:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors, return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;  
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)     
    else:
        sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        input_data = ST_exp.X
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors, return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;  
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors, k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)   
        
    return A_exp



def A_intra_transfer(data, data_type, real_num, pseudo_num):
    """
    将域内邻接矩阵扩展到真实斑点与伪斑点共同组成的大图中。

    参数:
        data: 单一域内部邻接矩阵。
        data_type: `real` 或 `pseudo`，决定填入大图的左上或右下块。
        real_num: 真实斑点数量。
        pseudo_num: 伪斑点数量。

    返回:
        ndarray: 大图尺寸的块对角邻接矩阵。
    """
    
    adj = np.zeros((real_num+pseudo_num, real_num+pseudo_num), dtype=float)
    if data_type == 'real':      
        adj[:real_num, :real_num] = data
    elif data_type == 'pseudo':
        adj[real_num:, real_num:] = data
        
    return adj



def adj_normalize(mx, symmetry=True):
    """
    对邻接矩阵进行度归一化。

    参数:
        mx: 输入邻接矩阵。
        symmetry: `True` 使用 `D^-1/2 A D^-1/2`，否则使用 `D^-1 A`。

    返回:
        matrix: 归一化后的稠密矩阵。
    """
    
    mx = sp.csr_matrix(mx)
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0. 
    if symmetry == True:
        r_mat_inv = sp.diags(np.sqrt(r_inv))
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    else:
        r_mat_inv = sp.diags(r_inv) 
        mx = r_mat_inv.dot(mx)
    
    return mx.todense()
