# -*- coding: utf-8 -*-
"""
自编码器降维模块。

该模块在 `data_integration` 中作为可选降维方法使用。自编码器学习一个
低维嵌入，使真实空间斑点和伪斑点可以在同一特征空间中进行建图或训练。
"""
import torch
import torch.nn as nn
import time
import scanpy as sc
import multiprocessing



def full_block(in_features, out_features, p_drop):
        """
        构建自编码器中的标准全连接块。

        每个块包含线性层、LayerNorm、ELU 激活和 Dropout，既能稳定训练，
        又能在小批量或全量矩阵训练时减轻特征尺度差异。
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.LayerNorm(out_features),
            nn.ELU(),
            nn.Dropout(p=p_drop),
        )

class autoencoder(nn.Module):
    """
    用于表达矩阵降维的自编码器（AutoEncoder）。

    编码器将高维基因表达压缩到低维嵌入，解码器尝试重构原始表达。
    训练完成后仅返回编码器输出，用作下游整合和建图特征。
    """
    def __init__(self, x_size, hidden_size, embedding_size, p_drop=0):
        """
        初始化编码器和解码器。

        参数:
            x_size: 原始输入特征维度。
            hidden_size: 中间隐藏层维度。
            embedding_size: 低维嵌入维度。
            p_drop: Dropout 概率。
        """
        super(autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            full_block(x_size, hidden_size, p_drop),
            full_block(hidden_size, embedding_size, p_drop)
        )
        
        self.decoder = nn.Sequential(
            full_block(embedding_size, hidden_size, p_drop),
            full_block(hidden_size, x_size, p_drop)
        )
        
    def forward(self, x):
        """
        前向传播并同时返回低维嵌入和重构结果。

        返回:
            tuple: `(embedding, reconstruction, modules)`。
        """
        
        en = self.encoder(x)
        de = self.decoder(en)
        
        return en, de, [self.encoder, self.decoder]

def auto_train(model, epoch_n, loss_fn, optimizer, data, cpu_num=-1, device='GPU'):
    """
    训练自编码器模型并返回编码后的低维特征。

    参数:
        model: 自编码器模型实例。
        epoch_n: 训练轮数。
        loss_fn: 重构损失函数，通常为 MSELoss。
        optimizer: PyTorch 优化器。
        data: 待降维的表达矩阵张量。
        cpu_num: CPU 线程数，`-1` 表示使用全部核心。
        device: `GPU` 时优先使用 CUDA，否则在 CPU 上运行。

    返回:
        Tensor: 迁回 CPU 的编码器输出。
    """
    
    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)
    
    if device == 'GPU':
        if torch.cuda.is_available():
            model = model.cuda()
            data = data.cuda()

    # 自编码器在这里使用全量矩阵训练；数据量较大时每轮前清空显存，
    # 可以降低与前一轮临时张量叠加造成的峰值占用。
    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        train_cost = 0       
            
        optimizer.zero_grad()
        en, de, _ = model(data)
        
        loss = loss_fn(de, data)
        
        loss.backward()
        optimizer.step()
    
    torch.cuda.empty_cache()
    
    return en.cpu()

