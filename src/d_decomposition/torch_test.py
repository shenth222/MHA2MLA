import torch
import sys
sys.path.append("/data/shenth/work/d_decomposition/src/")
from d_decomposition import d_decomposition_torch

matrix = torch.rand((4, 4))
p, d, q, a_hat, delta = d_decomposition_torch(matrix, 2)
from IPython import embed
embed()