# -*- coding: utf-8 -*-
"""
CKGC 图卷积与图数据构建工具。

CKGC（基于边位置编码的图卷积）在本项目中承担核心消息传递职责。
它将邻接矩阵转换为边索引与随机游走位置编码（RRWP），再根据边特征
动态生成多头消息权重。为了适配空间转录组的大图场景，默认使用分块
消息传播以降低 GPU 显存峰值。
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import Sequential
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter
# 兼容不同 PyG 版本的 SparseTensor 类型定义，当前实现未直接依赖该类型。
# from torch_sparse import SparseTensor
from torch_geometric.typing import SparseTensor
from timm.models.layers import trunc_normal_
from typing import Literal
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


class CKGConvParameters:
    """
    CKGConv 的参数容器。

    使用显式参数对象可以避免在模型构造时传入过长的参数列表，并便于
    首层和隐藏层复用同一组图卷积配置。
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            pe_dim,
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
    ):
        """
        保存图卷积结构、归一化、Dropout 和显存优化相关参数。

        参数:
            in_dim: 输入节点特征维度。
            out_dim: 每个节点输出特征维度。
            pe_dim: 边位置编码维度，即 RRWP 阶数。
            num_heads: 多头消息权重数量。
            clamp: 是否限制边权网络输出范围，`None` 表示不截断。
            act: MLP 激活函数类型。
            batch_norm: 是否在边权网络中使用 BatchNorm。
            layer_norm: 是否在边权网络中使用 LayerNorm。
            weight_norm: 是否对线性层启用 WeightNorm。
            deg_scaler: 是否根据节点度数修正聚合结果。
            ffn_ratio: 边权 MLP 隐藏层放大倍数。
            average: 是否对邻居消息取平均；否则执行求和聚合。
            num_blocks: 边权 MLP 中残差块数量。
            mlp_dropout: 边权 MLP Dropout 概率。
            attn_dropout: 边权/注意力 Dropout 概率。
            out_proj: 是否在多头拼接后投影回 `out_dim`。
            softmax: 是否对同一目标节点的边权做 softmax 归一化。
            softplus: 是否强制边权为正。
            value_proj: 保留参数，用于兼容历史配置。
            loss_memory: 是否启用分块低显存传播。
            chunk_size: 手动指定分块边数；为 `None` 时自动估计。
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pe_dim = pe_dim
        self.num_heads = num_heads
        self.clamp = clamp
        self.act = act
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.weight_norm = weight_norm
        self.deg_scaler = deg_scaler
        self.ffn_ratio = ffn_ratio
        self.average = average
        self.num_blocks = num_blocks
        self.mlp_dropout = mlp_dropout
        self.attn_dropout = attn_dropout
        self.out_proj = out_proj
        self.softmax = softmax
        self.softplus = softplus
        self.value_proj = value_proj
        self.loss_memory = loss_memory
        self.chunk_size = chunk_size


def trunc_init_(m):
    """对线性层、卷积层或参数张量执行截断正态初始化。"""
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        trunc_normal_(m.weight, std=0.1)
    elif isinstance(m, nn.Parameter):
        trunc_normal_(m, std=0.1)


def _wn_linear(l):
    """为线性层添加 WeightNorm 参数化。"""
    if isinstance(l, nn.Linear):
        nn.utils.parametrizations.weight_norm(l, name="weight")


class ResidualLayer(nn.Module):
    """
    通用残差连接层。

    默认执行 `x + x_res`。当启用 ReZero 或 LayerScale 时，会引入可学习
    缩放系数，帮助较深的边权网络在训练初期保持稳定。
    """

    def __init__(self, rezero=False, layerscale=False, alpha=0.1, dim=1):
        super().__init__()
        self.rezero = rezero
        self.layerscale = layerscale
        self.layerscale_init = alpha

        if rezero:
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif layerscale:
            self.alpha = nn.Parameter(torch.ones(1, dim) * alpha, requires_grad=True)
        else:
            self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)

        self.dim = self.alpha.size(-1)

    def forward(self, x, x_res):
        """合并当前分支输出与残差输入。"""
        if not self.rezero and not self.layerscale:
            return x + x_res

        return x * self.alpha + x_res

    def __repr__(self):
        return (
            f"{super().__repr__()}(rezero={self.rezero}, layerscale={self.layerscale}, "
            f"layerscale_init={self.layerscale_init}, "
            f"alpha_shape={self.alpha.shape})"
        )


class CKGConv(nn.Module):
    """
    基于边位置编码的多头图卷积层。

    `CKGConv` 不直接学习固定邻接权重，而是先把 RRWP 边特征输入到
    一个小型 MLP 中生成边级权重，再用该权重调制源节点特征并聚合到
    目标节点。这样可以让模型根据图结构上下文动态调整消息强度。
    """

    def __init__(self, ckgc_p: CKGConvParameters):
        """根据参数对象构建边权网络、多头投影和输出投影。"""
        super().__init__()

        self.in_dim = ckgc_p.in_dim
        self.out_dim = ckgc_p.out_dim
        self.pe_dim = ckgc_p.pe_dim
        self.num_heads = ckgc_p.num_heads
        self.clamp = np.abs(ckgc_p.clamp) if ckgc_p.clamp is not None else None

        self.batch_norm = ckgc_p.batch_norm
        self.layer_norm = ckgc_p.layer_norm
        self.deg_scaler = ckgc_p.deg_scaler
        self.softmax = ckgc_p.softmax
        self.softplus = ckgc_p.softplus
        self.weight_norm = ckgc_p.weight_norm

        self.ffn_ratio = ckgc_p.ffn_ratio
        self.average = ckgc_p.average
        self.kernel = None
        self.loss_memory = ckgc_p.loss_memory
        self.chunk_size = ckgc_p.chunk_size

        num_blocks = ckgc_p.num_blocks

        if self.batch_norm:
            norm_fn = nn.BatchNorm1d
        elif self.layer_norm:
            norm_fn = nn.LayerNorm
        else:
            norm_fn = nn.Identity

        self.mlp_dropout = ckgc_p.mlp_dropout

        # 边权网络以 RRWP 为输入，输出维度为 num_heads * out_dim；
        # 每个头都能为每个输出维度学习独立的边调制系数。
        blocks = []

        hid_dim = ckgc_p.out_dim * ckgc_p.num_heads
        if ckgc_p.pe_dim != ckgc_p.in_dim:
            blocks += [(nn.Linear(ckgc_p.pe_dim, hid_dim), "x -> x")]

        for i in range(num_blocks):
            blocks = blocks + [
                (norm_fn(hid_dim), "x -> h"),
                (ckgc_p.act(), "h -> h"),
                (
                    nn.Linear(hid_dim, int(hid_dim * self.ffn_ratio), bias=True),
                    "h -> h",
                ),
                (norm_fn(int(hid_dim * self.ffn_ratio)), "h -> h"),
                (ckgc_p.act(), "h -> h"),
                (
                    nn.Linear(int(hid_dim * self.ffn_ratio), hid_dim, bias=True),
                    "h -> h",
                ),
                (
                    (
                        nn.Dropout(self.mlp_dropout)
                        if self.mlp_dropout > 0
                        else nn.Identity()
                    ),
                    "h -> h",
                ),
                (
                    ResidualLayer(rezero=False, layerscale=False, dim=hid_dim),
                    "h, x -> x",
                ),
            ]

        blocks = blocks + [
            (norm_fn(hid_dim), "x -> x"),
            (nn.Linear(hid_dim, hid_dim, bias=True), "x -> x"),
        ]

        if self.softplus:
            blocks += [(nn.Softplus(), "x -> x")]

        self.attn_dropout = (
            nn.Dropout(ckgc_p.attn_dropout)
            if ckgc_p.attn_dropout > 0
            else nn.Identity()
        )

        self.blocks = Sequential("x", blocks)
        self.blocks.apply(trunc_init_)

        if self.weight_norm:
            self.blocks.apply(_wn_linear)

        self.out_proj = ckgc_p.out_proj
        if self.out_proj:
            self.out_layer = nn.Linear(hid_dim, ckgc_p.out_dim, bias=True)
        else:
            self.out_layer = nn.Identity()

        self.x_to_mult_head = nn.Linear(ckgc_p.in_dim, hid_dim, bias=True)
        self.bias = nn.Parameter(
            torch.zeros(1, ckgc_p.num_heads, ckgc_p.out_dim), requires_grad=True
        )
        self.x_to_mult_head.apply(trunc_init_)

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, hid_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

    def propagate_attention_loss_memory(self, x, pe_index, pe_val, deg=None):
        """
        低显存分块消息传播。

        大规模空间转录组数据的边数可能较多，若一次性计算所有边权会造成
        显存峰值过高。该实现按边分块计算权重和消息，并在保持计算图的
        前提下累加到目标节点。

        参数:
            x: 已投影为多头形状的节点特征，形状为 `[N, heads, out_dim]`。
            pe_index: 边索引，形状为 `[2, E]`，第 0 行为目标节点。
            pe_val: 每条边的 RRWP 特征，形状为 `[E, pe_dim]`。
            deg: 节点度数，用于平均聚合或度数缩放。

        返回:
            Tensor: 聚合后的节点表示，形状为 `[N, heads * out_dim]`。
        """
        edge_index = pe_index
        E = pe_val
        num_nodes = x.size(0)
        reduce = "mean" if self.average else "add"
        if self.chunk_size is None:
            self.chunk_size = self._calculate_optimal_chunk_size(x)  # 动态计算分块大小

        # 预分配聚合结果。这里使用加法产生新张量而不是原地写入，
        # 目的是保持 autograd 能够追踪每个分块对最终损失的贡献。
        wV = torch.zeros(
            num_nodes,
            self.num_heads,
            self.out_dim,
            dtype=torch.float32,  # 确保最终精度一致
            device=x.device,
            requires_grad=True,
        )
        max_values = None
        sum_exp = None

        # 若启用 softmax，需要跨所有分块得到每个目标节点的全局归一化因子。
        # 因此先做两遍扫描：第一遍求最大值，第二遍求指数和。
        if self.softmax:
            # 第一阶段：全局最大值
            max_values = torch.full(
                (num_nodes,), -float("inf"), dtype=E.dtype, device=E.device
            )
            for i in range(0, edge_index.size(1), self.chunk_size):
                chunk_slice = slice(i, i + self.chunk_size)
                chunk_edges = edge_index[:, chunk_slice]
                chunk_E = E[chunk_slice]

                chunk_score = self.blocks(chunk_E)
                if self.clamp is not None:
                    chunk_score = chunk_score.clamp(min=-self.clamp, max=self.clamp)

                current_max = scatter(
                    chunk_score, chunk_edges[0], dim=0, dim_size=num_nodes, reduce='max'
                )
                max_values = torch.maximum(max_values, current_max)

                # 及时清理
                del chunk_score, current_max, chunk_edges, chunk_E

            # 第二阶段：全局指数和
            sum_exp = torch.zeros_like(max_values)
            for i in range(0, edge_index.size(1), self.chunk_size):
                chunk_slice = slice(i, i + self.chunk_size)
                chunk_edges = edge_index[:, chunk_slice]
                chunk_E = E[chunk_slice]

                chunk_score = self.blocks(chunk_E)
                if self.clamp is not None:
                    chunk_score = chunk_score.clamp(min=-self.clamp, max=self.clamp)

                chunk_exp = (chunk_score - max_values[chunk_edges[0]]).exp()
                sum_exp.scatter_add_(0, chunk_edges[0], chunk_exp)

                del chunk_score, chunk_exp, chunk_edges, chunk_E

        # 主处理循环：分块计算边权、生成消息，并按目标节点聚合。
        for i in range(0, edge_index.size(1), self.chunk_size):
            chunk_slice = slice(i, i + self.chunk_size)
            chunk_edges = edge_index[:, chunk_slice]
            chunk_E = E[chunk_slice]

            # 分块计算边权网络，避免一次性 materialize 全量边权。
            chunk_score = self.blocks(chunk_E)

            # 可选截断用于限制边权范围，防止指数或乘法放大导致数值不稳定。
            if self.clamp is not None:
                chunk_score = chunk_score.clamp(min=-self.clamp, max=self.clamp)

            # Softmax调整
            if self.softmax:
                chunk_max = max_values[chunk_edges[0]]
                chunk_exp = (chunk_score - chunk_max).exp()
                chunk_sum_exp = sum_exp[chunk_edges[0]] + 1e-8
                chunk_score = chunk_exp / chunk_sum_exp

                del chunk_max, chunk_exp, chunk_sum_exp

            # Dropout 作用在边权上，相当于随机弱化部分边的消息。
            chunk_score = self.attn_dropout(chunk_score)
            chunk_score = chunk_score.view(-1, self.num_heads, self.out_dim)

            # 消息计算与聚合：chunk_edges[1] 为源节点，chunk_edges[0] 为目标节点。
            chunk_msg = x[chunk_edges[1]] * chunk_score
            # 修正要点：强制使用 add 聚合，避免每个分块单独取平均导致数值偏差。
            contribution = scatter(
                chunk_msg, chunk_edges[0], dim=0, dim_size=num_nodes, reduce="add"
            ).to(
                torch.float32
            )  # 确保精度

            # 累加方式保持计算图
            wV = wV + contribution

            # --- 显存清理（安全操作）---
            del chunk_msg, contribution, chunk_score, chunk_edges, chunk_E

        del max_values, sum_exp

        # 平均聚合修正：分块阶段统一使用 add 聚合，避免每个块单独平均造成偏差。
        # 只有未启用 softmax 时才除以度数，因为 softmax 已经完成归一化。
        if self.average and not self.softmax and deg is not None:
             wV = wV / deg.to(wV.device).view(-1, 1, 1).clamp(min=1e-8)

        # 可选度数缩放，让模型学习高/低度节点的不同响应。
        if self.deg_scaler and deg is not None:
            sqrt_deg = torch.sqrt(deg.to(torch.float32)).view(-1, 1)
            h = wV.view(num_nodes, -1)
            h = h * self.deg_coef[..., 0] + h * sqrt_deg * self.deg_coef[..., 1]
            wV = h.view_as(wV)

            del sqrt_deg, h

        return (wV + self.bias).view(wV.size(0), -1)

    def _calculate_optimal_chunk_size(self, x):
        """
        根据设备可用内存估算每个分块处理的边数。

        GPU 场景下保守使用约 80% 空闲显存，并设置上下限，避免小图切得太碎
        或大图一次性占用过多显存。CPU 场景使用固定默认值。
        """
        if x.is_cuda:
            try:
                free_mem = torch.cuda.mem_get_info()[0]  # 剩余显存
                element_size = 4 if x.dtype == torch.float32 else 2
                per_element = self.num_heads * self.out_dim * element_size
                safe_size = int((free_mem * 0.8) // (per_element * 4))  # 安全系数
                return max(1000, min(safe_size, 50000))  # 上限5万边/块
            except Exception:
                return 10000
        else:
            # CPU模式下使用默认分块
            return 10000

    def propagate_attention(self, x, pe_index, pe_val, deg=None):
        """
        一次性消息传播实现。

        该路径逻辑更直接，适合小图或显存充足的场景。默认训练使用
        `propagate_attention_loss_memory`，以优先保证大图稳定运行。
        """
        edge_index = pe_index
        E = pe_val

        E = self.blocks(E)
        score = E
        reduce = "mean" if self.average else "add"

        if self.clamp is not None:
            # 对数据进行截断
            score = torch.clamp(score, min=-self.clamp.abs(), max=self.clamp.abs())

        if self.softmax:
            reduce = "add"
            score = pyg_softmax(score, edge_index[0], num_nodes=x.size(0), dim=0)

        self.kernel = score
        score = self.attn_dropout(score).view(-1, self.num_heads, self.out_dim)
        msg = x[edge_index[1]] * score

        wV = scatter(msg, edge_index[0], dim=0, dim_size=x.size(0), reduce=reduce)

        if self.deg_scaler and deg is not None:
            sqrt_deg = torch.sqrt(deg)
            sqrt_deg = sqrt_deg.view(x.size(0), 1)
            h = wV.view(wV.size(0), -1)
            h = h * self.deg_coef[:, :, 0] + h * sqrt_deg * self.deg_coef[:, :, 1]
            wV = h.view(-1, self.num_heads, self.out_dim)

        return (wV + self.bias).view(wV.size(0), -1)

    def forward(self, x, pe_index, pe_val, deg):
        """
        投影节点特征、执行图消息传播，并输出节点表示。

        参数:
            x: 输入节点特征，形状 `[N, in_dim]`。
            pe_index: 图边索引。
            pe_val: 边位置编码。
            deg: 节点度数。

        返回:
            Tensor: 图卷积后的节点特征，形状 `[N, out_dim]`。
        """
        h = self.x_to_mult_head(x).view(-1, self.num_heads, self.out_dim)

        wV = (
            self.propagate_attention_loss_memory(h, pe_index, pe_val, deg)
            if self.loss_memory
            else self.propagate_attention(h, pe_index, pe_val, deg)
        )

        h_out = self.out_layer(wV)

        return h_out


class GraphDataBuilder:
    """
    将邻接矩阵或边表转换为 CKGC 所需的图输入。

    输出的每个图都包含三部分：边索引、边级 RRWP 特征和节点度数。RRWP
    用于描述一条边在多阶随机游走中的结构角色，是 CKGC 动态边权网络
    的主要输入。
    """

    def __init__(
            self,
            df_dict,
            node_num=None,
            pe_dim=6,
            if_weight=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        """
        构建一个或多个图类型的 CKGC 输入。

        参数:
            df_dict: 图类型到图数据的映射。值可以是方阵 Tensor，或包含
                `Cell1`、`Cell2` 列的 DataFrame。
            node_num: 图中节点总数；为 `None` 时从输入图中自动推断。
            pe_dim: RRWP 编码维度，也就是随机游走阶数。
            if_weight: 是否使用输入边权；关闭时所有边权视为 1。
            device: 计算 RRWP 时使用的设备。
        """
        # 确定所有图类型中的最大节点索引，保证不同图共享同一节点空间。
        max_node = 0
        for df in df_dict.values():
            if isinstance(df, pd.DataFrame):
                max_node = max(max_node, df["Cell1"].max(), df["Cell2"].max())
            elif torch.is_tensor(df):
                max_node = max(max_node, df.shape[0] - 1)
            else:
                raise TypeError(f"Unsupported type in df_dict: {type(df)}")

        self.node_num = max_node + 1 if node_num is None else node_num

        self.graphs = {}
        for gtype, graph_data in df_dict.items():
            # Tensor 输入表示完整邻接矩阵，非零元素即为边。
            if torch.is_tensor(graph_data):
                assert graph_data.dim() == 2 and graph_data.size(0) == graph_data.size(1), \
                    "Tensor must be square adjacency matrix"
                assert graph_data.size(0) <= self.node_num, \
                    f"Adjacency matrix size {graph_data.size(0)} exceeds node_num {self.node_num}"

                # 提取邻接矩阵的边索引，转成 PyG 风格 `[2, E]` 表示。
                indices = torch.nonzero(graph_data != 0).t()
                row, col = indices[0], indices[1]

                # 如果没有权重，默认设为 1
                if if_weight:
                    edge_attr = graph_data[row, col].float()
                else:
                    edge_attr = torch.ones(row.size(0), dtype=torch.float)

            # DataFrame 输入兼容旧版本边表，列名沿用 Cell1/Cell2。
            elif isinstance(graph_data, pd.DataFrame):
                row = torch.LongTensor(graph_data["Cell1"].values)
                col = torch.LongTensor(graph_data["Cell2"].values)

                # 节点索引校验
                assert (row.max() < self.node_num and col.max() < self.node_num), \
                    f"图类型 {gtype} 中的节点索引超出了 {self.node_num - 1}"

                if "Distance" in graph_data.columns and if_weight:
                    edge_attr = torch.FloatTensor(graph_data["Distance"].values)
                else:
                    edge_attr = torch.ones(len(graph_data), dtype=torch.float)

            else:
                raise TypeError(f"Unsupported type for graph data: {type(graph_data)}")

            edge_index = torch.stack([row, col], dim=0)
            rrwp, deg = self._compute_rrwp(
                edge_index,
                edge_attr,
                pe_dim,
                self.node_num,
                device,
            )
            self.graphs[gtype] = [edge_index.to("cpu"), rrwp.to("cpu"), deg.to("cpu")]
            del rrwp, deg, edge_index
            if "cuda" in str(device):
                torch.cuda.empty_cache()

    def _compute_rrwp(self, edge_index, edge_attr, pe_dim, node_num, device):
        """
        计算边级相对随机游走位置编码（RRWP）。

        RRWP 的第 k 维表示从边的源/目标节点沿随机游走 k 步后仍落在原始
        边位置上的概率强度。这里会用原始边掩码过滤非输入边，确保输出
        特征与 `edge_index` 一一对应。
        """
        # 获取原始边的行列索引
        row = edge_index[0].to(device)
        col = edge_index[1].to(device)
        num_edges = row.size(0)  # 原始边的数量

        # 构建稠密邻接矩阵后计算随机游走矩阵。当前实现偏重可读性；
        # 若节点数继续扩大，可以考虑替换为稀疏矩阵乘法。
        dense = torch.zeros((node_num, node_num), device=device)
        dense[row, col] = edge_attr.to(device)
        dense_bool = (dense > 0).to(torch.bool)  # 原始边的布尔掩码
        deg = (dense > 0).sum(dim=1, keepdim=True)  # 节点度数
        rw = dense / (deg.view(-1, 1) + 1e-8)  # 归一化的随机游走矩阵
        out1 = rw.clone()
        rrwp = []

        for i in range(pe_dim):
            # 应用布尔掩码过滤非原始边
            out1_filtered = out1 * dense_bool

            # 直接通过原始边索引获取特征值（包含零值）
            current_values = out1_filtered[row, col]

            # 收集特征值
            rrwp.append(current_values.to("cpu"))

            # 提前终止最后一次循环的矩阵乘法
            if i < pe_dim - 1:
                out1 = out1_filtered @ rw  # 矩阵乘法保持路径连续性

        # 堆叠所有特征维度
        rrwp_tensor = torch.stack(rrwp, dim=1)

        return rrwp_tensor, deg.to("cpu")

    def get_graphs(self):
        """返回已构建好的图输入字典。"""
        return self.graphs


if __name__ == "__main__":
    x = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

    adj = torch.Tensor([[0, 1, 0],
                        [0, 1, 1],
                        [0, 1, 0]])
    # df = pd.DataFrame(adj.T, columns=["Cell1", "Cell2"])
    g = GraphDataBuilder({1: adj, 2: adj, 3: adj}, 3, 5).get_graphs()

    CKGConv_p = CKGConvParameters(
        in_dim=3,
        out_dim=3,
        pe_dim=5,
    )
    CKGC = CKGConv(CKGConv_p).to("cuda:0")

    x = torch.FloatTensor(x).to("cuda:0")

    print(
        CKGC(x, *[tensor.to("cuda:0") for tensor in g[1]])
    )
    pass
