import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import Sequential
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter
# from torch_sparse import SparseTensor
from torch_geometric.typing import SparseTensor
from timm.models.layers import trunc_normal_
from typing import Literal
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.functional as F


class CKGConvParameters:
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
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        trunc_normal_(m.weight, std=0.1)
    elif isinstance(m, nn.Parameter):
        trunc_normal_(m, std=0.1)


def _wn_linear(l):
    if isinstance(l, nn.Linear):
        nn.utils.parametrizations.weight_norm(l, name="weight")


class ResidualLayer(nn.Module):
    """
    残差层 (Residual Layer);
    - 支持 rezero 和 layerscale 机制
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
    简化版 CKGConv
    > 暂不包含 deg-scaler
    """

    def __init__(self, ckgc_p: CKGConvParameters):
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
        edge_index = pe_index
        E = pe_val
        num_nodes = x.size(0)
        reduce = "mean" if self.average else "add"
        if self.chunk_size is None:
            self.chunk_size = self._calculate_optimal_chunk_size(x)  # 动态计算分块大小

        # ==== 预分配结果张量 ====
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

        # ==== Softmax全局统计量计算 ====
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

        # ==== 主处理循环 ====
        for i in range(0, edge_index.size(1), self.chunk_size):
            chunk_slice = slice(i, i + self.chunk_size)
            chunk_edges = edge_index[:, chunk_slice]
            chunk_E = E[chunk_slice]

            # --- 分块计算核心 ---
            chunk_score = self.blocks(chunk_E)

            # Clamp处理
            if self.clamp is not None:
                chunk_score = chunk_score.clamp(min=-self.clamp, max=self.clamp)

            # Softmax调整
            if self.softmax:
                chunk_max = max_values[chunk_edges[0]]
                chunk_exp = (chunk_score - chunk_max).exp()
                chunk_sum_exp = sum_exp[chunk_edges[0]] + 1e-8
                chunk_score = chunk_exp / chunk_sum_exp

                del chunk_max, chunk_exp, chunk_sum_exp

            # Dropout处理（保持随机一致性）
            chunk_score = self.attn_dropout(chunk_score)
            chunk_score = chunk_score.view(-1, self.num_heads, self.out_dim)

            # 消息计算与聚合
            chunk_msg = x[chunk_edges[1]] * chunk_score
            # FIX: 强制使用'add'聚合，避免分块平均导致数值错误
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

        # ==== 后处理：平均值修正 ====
        # 只有在average=True且未启用softmax时（因为softmax隐含了归一化），才除以度数
        if self.average and not self.softmax and deg is not None:
             wV = wV / deg.to(wV.device).view(-1, 1, 1).clamp(min=1e-8)

        # ==== 后处理 ====
        if self.deg_scaler and deg is not None:
            sqrt_deg = torch.sqrt(deg.to(torch.float32)).view(-1, 1)
            h = wV.view(num_nodes, -1)
            h = h * self.deg_coef[..., 0] + h * sqrt_deg * self.deg_coef[..., 1]
            wV = h.view_as(wV)

            del sqrt_deg, h

        return (wV + self.bias).view(wV.size(0), -1)

    def _calculate_optimal_chunk_size(self, x):
        """动态计算分块大小"""
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
        h = self.x_to_mult_head(x).view(-1, self.num_heads, self.out_dim)

        wV = (
            self.propagate_attention_loss_memory(h, pe_index, pe_val, deg)
            if self.loss_memory
            else self.propagate_attention(h, pe_index, pe_val, deg)
        )

        h_out = self.out_layer(wV)

        return h_out


class GraphDataBuilder:
    def __init__(
            self,
            df_dict,
            node_num=None,
            pe_dim=6,
            if_weight=False,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ):
        # 确定所有图类型中的最大节点索引
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
            # 处理 Tensor 输入 (邻接矩阵)
            if torch.is_tensor(graph_data):
                assert graph_data.dim() == 2 and graph_data.size(0) == graph_data.size(1), \
                    "Tensor must be square adjacency matrix"
                assert graph_data.size(0) <= self.node_num, \
                    f"Adjacency matrix size {graph_data.size(0)} exceeds node_num {self.node_num}"

                # 提取邻接矩阵的边索引
                indices = torch.nonzero(graph_data != 0).t()
                row, col = indices[0], indices[1]

                # 如果没有权重，默认设为 1
                if if_weight:
                    edge_attr = graph_data[row, col].float()
                else:
                    edge_attr = torch.ones(row.size(0), dtype=torch.float)

            # 处理 DataFrame 输入 (兼容旧版本行为)
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
        # 获取原始边的行列索引
        row = edge_index[0].to(device)
        col = edge_index[1].to(device)
        num_edges = row.size(0)  # 原始边的数量

        # 构建稀疏邻接矩阵
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
