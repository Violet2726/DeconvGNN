"""
STdGCN 核心算法包
包含数据预处理、图构建、GCN 模型定义及训练预测流程。
"""
from ._version import __version__
from .utils import *
from .autoencoder import *
from .adjacency_matrix import *
from .GCN import *

from .STdGCN import *