# -*- coding: utf-8 -*-
"""
STdGCN 图神经网络模型与训练流程。

本模块定义了双通道 GNN：表达相似图通道和空间邻近图通道分别编码节点，
再将两个通道的表示拼接后送入全连接层输出细胞类型比例。训练阶段只使用
伪斑点标签进行监督，真实空间斑点用于最终推理。
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import time
import multiprocessing
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy

from core.CKGC import CKGConvParameters, CKGConv, GraphDataBuilder


class conGraphConvolutionlayer(Module):
    """
    传统 GCN 图卷积层。

    该层是早期实现的保留版本，目前主模型已改用 `CKGConv`。保留它可以
    方便后续做消融实验或回退到稀疏邻接矩阵乘法版本。
    """

    def __init__(self, in_features, out_features, bias=True):
        """初始化线性投影权重和可选偏置。"""
        super(conGraphConvolutionlayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """按输出维度初始化权重，保持与常见 GCN 实现一致。"""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """执行 `A @ (X @ W)` 的图卷积计算。"""
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
                self.__class__.__name__
                + " ("
                + str(self.in_features)
                + " -> "
                + str(self.out_features)
                + ")"
        )


class conGCN(nn.Module):
    """
    STdGCN 主模型。

    模型包含两条结构相同但权重独立的图通道：
    - 表达图通道建模真实斑点与伪斑点之间、以及各自内部的表达相似性；
    - 空间图通道建模真实空间斑点的物理邻近关系。

    两个通道的节点表示拼接后进入全连接分类/比例预测头，最终通过
    `log_softmax` 输出适配 KL 散度损失的对数概率。
    """

    def __init__(
            self,
            nfeat,
            nhid,
            common_hid_layers_num,
            fcnn_hid_layers_num,
            dropout,
            nout1,
    ):
        """
        构建双通道 GNN 网络。

        参数:
            nfeat: 输入节点特征维度。
            nhid: 图卷积隐藏层维度。
            common_hid_layers_num: 输入层之后额外堆叠的图卷积层数。
            fcnn_hid_layers_num: 输出头中额外的全连接隐藏层数。
            dropout: Dropout 概率。
            nout1: 输出细胞类型数量。
        """
        super(conGCN, self).__init__()

        self.nfeat = nfeat
        self.nhid = nhid
        self.common_hid_layers_num = common_hid_layers_num
        self.fcnn_hid_layers_num = fcnn_hid_layers_num
        self.nout1 = nout1
        self.dropout = dropout
        self.training = True

        CKGConv_p = CKGConvParameters(
            in_dim=nfeat,
            out_dim=nhid,
            pe_dim=24,
            num_heads=3,
            clamp=None,
            act=nn.GELU,
            batch_norm=False,
            layer_norm=False,
            weight_norm=False,
            deg_scaler=False,
            ffn_ratio=1.0,
            average=True,
            num_blocks=1,
            mlp_dropout=0.0,
            attn_dropout=0.0,
            out_proj=True,
            softmax=False,
            softplus=False,
            value_proj="mean",
            loss_memory=True,
            chunk_size=None,
        )

        # 首层图卷积分别处理表达图和空间图。两个通道不共享参数，
        # 使模型可以学习“表达相似”和“空间邻近”两类关系的不同权重。
        self.gc_in_exp = CKGConv(CKGConv_p)
        self.bn_node_in_exp = nn.BatchNorm1d(nhid)

        self.gc_in_sp = CKGConv(CKGConv_p)
        self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        # 额外图卷积层使用动态属性命名，保持与历史参数配置兼容。
        # 注意：这里保留 `exec` 是为了不改变既有模型参数命名和存档格式。
        if self.common_hid_layers_num > 0:
            CKGConv_p_hid = CKGConvParameters(
                in_dim=nhid,
                out_dim=nhid,
                pe_dim=24,
                num_heads=3,
                clamp=None,
                act=nn.GELU,
                batch_norm=False,
                layer_norm=False,
                weight_norm=False,
                deg_scaler=False,
                ffn_ratio=1.0,
                average=True,
                num_blocks=1,
                mlp_dropout=0.0,
                attn_dropout=0.0,
                out_proj=True,
                softmax=False,
                softplus=False,
                value_proj="mean",
                loss_memory=True,
                chunk_size=None,
            )
            for i in range(self.common_hid_layers_num):
                exec("self.cgc{}_exp = CKGConv(CKGConv_p_hid)".format(i + 1))
                exec("self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)".format(i + 1))

                exec("self.cgc{}_sp = CKGConv(CKGConv_p_hid)".format(i + 1))
                exec("self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)".format(i + 1))

        # ## 旧版输入层实现：使用传统 GCN 稀疏矩阵乘法。
        # self.gc_in_exp = conGraphConvolutionlayer(nfeat, nhid)
        # self.bn_node_in_exp = nn.BatchNorm1d(nhid)
        # self.gc_in_sp = conGraphConvolutionlayer(nfeat, nhid)
        # self.bn_node_in_sp = nn.BatchNorm1d(nhid)

        # ## 旧版公共隐藏层实现。
        # if self.common_hid_layers_num > 0:
        #     for i in range(self.common_hid_layers_num):
        #         exec(
        #             "self.cgc{}_exp = conGraphConvolutionlayer(nhid, nhid)".format(
        #                 i + 1
        #             )
        #         )
        #         exec("self.bn_node_chid{}_exp = nn.BatchNorm1d(nhid)".format(i + 1))
        #         exec(
        #             "self.cgc{}_sp = conGraphConvolutionlayer(nhid, nhid)".format(i + 1)
        #         )
        #         exec("self.bn_node_chid{}_sp = nn.BatchNorm1d(nhid)".format(i + 1))

        # 输出头先拼接两个图通道表示，再映射为细胞类型比例分布。
        self.gc_out11 = nn.Linear(2 * nhid, nhid, bias=True)
        self.bn_out1 = nn.BatchNorm1d(nhid)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.gc_out11{} = nn.Linear(nhid, nhid, bias=True)".format(i + 1))
                exec("self.bn_out11{} = nn.BatchNorm1d(nhid)".format(i + 1))
        self.gc_out12 = nn.Linear(nhid, nout1, bias=True)

    def forward(self, x, adjs, graphs):
        """
        执行一次前向传播。

        参数:
            x: 所有节点的特征矩阵，顺序为真实斑点在前、伪斑点在后。
            adjs: 兼容旧实现的邻接矩阵列表，当前 CKGC 路径不直接使用。
            graphs: `GraphDataBuilder` 生成的图结构，包含边索引、RRWP 特征和度数。

        返回:
            tuple: `(log_probs, gc_list)`，其中 `log_probs` 为每个节点的
            细胞类型对数概率，`gc_list` 保存当前前向路径中的主要层引用。
        """

        self.x = x

        # 输入层：两个图通道分别编码同一批节点特征。
        self.x_exp = self.gc_in_exp(self.x, *graphs[1])
        self.x_exp = self.bn_node_in_exp(self.x_exp)
        self.x_exp = F.elu(self.x_exp)
        self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        self.x_sp = self.gc_in_sp(self.x, *graphs[2])
        self.x_sp = self.bn_node_in_sp(self.x_sp)
        self.x_sp = F.elu(self.x_sp)
        self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # 公共隐藏层：保持通道独立，避免空间关系直接覆盖表达关系。
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("self.x_exp = self.cgc{}_exp(self.x_exp, *graphs[1])".format(i + 1))
                exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i + 1))
                self.x_exp = F.elu(self.x_exp)
                self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
                exec("self.x_sp = self.cgc{}_sp(self.x_sp, *graphs[2])".format(i + 1))
                exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i + 1))
                self.x_sp = F.elu(self.x_sp)
                self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # ## 旧版前向传播：输入层。
        # self.x_exp = self.gc_in_exp(self.x, adjs[0])
        # self.x_exp = self.bn_node_in_exp(self.x_exp)
        # self.x_exp = F.elu(self.x_exp)
        # self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        # self.x_sp = self.gc_in_sp(self.x, adjs[1])
        # self.x_sp = self.bn_node_in_sp(self.x_sp)
        # self.x_sp = F.elu(self.x_sp)
        # self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # ## 旧版前向传播：公共隐藏层。
        # if self.common_hid_layers_num > 0:
        #     for i in range(self.common_hid_layers_num):
        #         exec("self.x_exp = self.cgc{}_exp(self.x_exp, adjs[0])".format(i + 1))
        #         exec("self.x_exp = self.bn_node_chid{}_exp(self.x_exp)".format(i + 1))
        #         self.x_exp = F.elu(self.x_exp)
        #         self.x_exp = F.dropout(self.x_exp, self.dropout, training=self.training)
        #         exec("self.x_sp = self.cgc{}_sp(self.x_sp, adjs[1])".format(i + 1))
        #         exec("self.x_sp = self.bn_node_chid{}_sp(self.x_sp)".format(i + 1))
        #         self.x_sp = F.elu(self.x_sp)
        #         self.x_sp = F.dropout(self.x_sp, self.dropout, training=self.training)

        # 输出头：拼接后的表示进入 MLP，输出各细胞类型的 log-probability。
        self.x1 = torch.cat([self.x_exp, self.x_sp], dim=1)
        self.x1 = self.gc_out11(self.x1)
        self.x1 = self.bn_out1(self.x1)
        self.x1 = F.elu(self.x1)
        self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        if self.fcnn_hid_layers_num > 0:
            for i in range(self.fcnn_hid_layers_num):
                exec("self.x1 = self.gc_out11{}(self.x1)".format(i + 1))
                exec("self.x1 = self.bn_out11{}(self.x1)".format(i + 1))
                self.x1 = F.elu(self.x1)
                self.x1 = F.dropout(self.x1, self.dropout, training=self.training)
        self.x1 = self.gc_out12(self.x1)

        # 返回层引用主要用于早停时深拷贝最佳参数，保持历史训练逻辑不变。
        gc_list = {}
        gc_list["gc_in_exp"] = self.gc_in_exp
        gc_list["gc_in_sp"] = self.gc_in_sp
        if self.common_hid_layers_num > 0:
            for i in range(self.common_hid_layers_num):
                exec("gc_list['cgc{}_exp'] = self.cgc{}_exp".format(i + 1, i + 1))
                exec("gc_list['cgc{}_sp'] = self.cgc{}_sp".format(i + 1, i + 1))
        gc_list["gc_out11"] = self.gc_out11
        if self.fcnn_hid_layers_num > 0:
            exec("gc_list['gc_out11{}'] =  self.gc_out11{}".format(i + 1, i + 1))
        gc_list["gc_out12"] = self.gc_out12

        return F.log_softmax(self.x1, dim=1), gc_list


def conGCN_train(
        model,
        train_valid_len,
        test_len,
        feature,
        adjs,
        label,
        epoch_n,
        loss_fn,
        optimizer,
        train_valid_ratio=0.9,
        scheduler=None,
        early_stopping_patience=5,
        clip_grad_max_norm=1,
        load_test_groundtruth=False,
        print_epoch_step=1,
        cpu_num=-1,
        GCN_device="CPU",
):
    """
    训练 STdGCN 模型并返回所有节点的预测结果。

    参数:
        model: 待训练的 `conGCN` 实例。
        train_valid_len: 伪斑点数量，训练/验证集均从该段中切分。
        test_len: 真实空间斑点数量，位于特征矩阵前半段。
        feature: GNN 输入节点特征矩阵。
        adjs: 表达图和空间图邻接矩阵，供 `GraphDataBuilder` 转换。
        label: 真实斑点占位标签和伪斑点监督标签拼接后的标签矩阵。
        epoch_n: 最大训练轮数。
        loss_fn: 训练损失函数，默认流程使用 KLDivLoss。
        optimizer: PyTorch 优化器。
        train_valid_ratio: 伪斑点中用于训练的比例，其余用于验证。
        scheduler: 可选学习率调度器。
        early_stopping_patience: 验证损失连续未改善的提前停止轮数。
        clip_grad_max_norm: 梯度裁剪阈值，防止训练早期梯度爆炸。
        load_test_groundtruth: 是否额外计算真实斑点测试损失。
        print_epoch_step: 日志打印间隔。
        cpu_num: CPU 线程数，`-1` 表示使用全部核心。
        GCN_device: 训练设备，`CPU` 或 `GPU`。

    返回:
        tuple: `(output, loss_history, trained_model)`，其中 `output` 已迁回 CPU。
    """
    if GCN_device == "CPU":
        device = torch.device("cpu")
        print("使用 CPU 进行计算。")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("使用 GPU 进行计算。")
        else:
            device = torch.device("cpu")
            print("使用 CPU 进行计算。")

    if cpu_num == -1:
        cores = multiprocessing.cpu_count()
        torch.set_num_threads(cores)
    else:
        torch.set_num_threads(cpu_num)


    # CKGC 需要边级随机游走位置编码，因此先将邻接矩阵转换为图数据。
    g_builder = GraphDataBuilder({1: adjs[0], 2: adjs[1]}, adjs[1].size(0), 24)
    g = g_builder.get_graphs()


    model = model.to(device)
    # adjs = [adj.to(device) for adj in adjs]
    feature = feature.to(device)
    label = label.to(device)

    time_open = time.time()

    # 特征矩阵顺序为 [真实斑点, 伪斑点]，训练索引需要整体后移 test_len。
    train_idx = range(int(train_valid_len * train_valid_ratio))
    valid_idx = range(len(train_idx), train_valid_len)

    best_val = np.inf
    clip = 0
    loss = []
    para_list = []

    # 图结构通常比模型小，但每轮重复迁移会浪费时间，因此训练前统一放到目标设备。
    for key in g.keys():
        for i in range(len(g[key])):
            if g[key][i] is not None:
                g[key][i] = g[key][i].to(device)

    for epoch in range(epoch_n):
        try:
            torch.cuda.empty_cache()
        except:
            pass

        optimizer.zero_grad()
        output1, paras = model(feature.float(), adjs, g)

        # 只使用伪斑点的标签训练。真实斑点的预测会在输出前保留，
        # 但默认不参与反向传播，避免把未知比例当成监督信号。
        loss_train1 = loss_fn(
            output1[list(np.array(train_idx) + test_len)],
            label[list(np.array(train_idx) + test_len)].float(),
        )
        loss_val1 = loss_fn(
            output1[list(np.array(valid_idx) + test_len)],
            label[list(np.array(valid_idx) + test_len)].float(),
        )
        if load_test_groundtruth == True:
            loss_test1 = loss_fn(output1[:test_len], label[:test_len].float())
            loss.append([loss_train1.item(), loss_val1.item(), loss_test1.item()])
        else:
            loss.append([loss_train1.item(), loss_val1.item(), None])

        if epoch % print_epoch_step == 0:
            print("******************************************")
            print(
                "轮数 {}/{}".format(epoch + 1, epoch_n),
                "训练集损失: {:.4f}".format(loss_train1.item()),
                "验证集损失: {:.4f}".format(loss_val1.item()),
                end="\t",
            )
            if load_test_groundtruth == True:
                print("测试集损失= {:.4f}".format(loss_test1.item()), end="\t")
            print("用时: {:.4f}s".format(time.time() - time_open))
        para_list.append(paras.copy())
        for i in paras.keys():
            para_list[-1][i] = copy.deepcopy(para_list[-1][i])

        # 早停基于四舍五入后的验证损失，降低浮点微小波动导致的误判。
        if early_stopping_patience > 0:
            if torch.round(loss_val1, decimals=4) < best_val:
                best_val = torch.round(loss_val1, decimals=4)
                best_paras = paras.copy()
                best_loss = loss.copy()
                clip = 1
                for i in paras.keys():
                    best_paras[i] = copy.deepcopy(best_paras[i])
            else:
                clip += 1
                if clip == early_stopping_patience:
                    break
        else:
            best_loss = loss.copy()
            best_paras = None

        # 训练更新：先反向传播，再裁剪梯度，最后执行优化器和学习率调度器。
        loss_train1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
        optimizer.step()
        if scheduler != None:
            try:
                scheduler.step()
            except:
                scheduler.step(metrics=loss_val1)

    print("*********************** 最终损失 (Final Loss) ***********************")
    print(
        "轮数 {}/{}".format(epoch + 1, epoch_n),
        "训练集损失: {:.4f}".format(loss_train1.item()),
        "验证集损失: {:.4f}".format(loss_val1.item()),
        end="\t",
    )
    if load_test_groundtruth == True:
        print("测试集损失= {:.4f}".format(loss_test1.item()), end="\t")
    print("用时: {:.4f}s".format(time.time() - time_open))

    torch.cuda.empty_cache()

    return output1.cpu(), loss, model.cpu()
