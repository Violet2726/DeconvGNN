# -*- coding: utf-8 -*-
"""
STdGCN 核心算法包。

对外暴露数据预处理、邻接图构建、自编码器、GNN 模型以及完整训练入口。
当前保持通配导入是为了兼容历史教程脚本中的调用方式。
"""
from ._version import __version__
from .utils import *
from .autoencoder import *
from .adjacency_matrix import *
from .GCN import *

from .STdGCN import *
