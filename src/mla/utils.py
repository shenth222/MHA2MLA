import torch
import torch.nn.functional as F

def apply_activation(x:torch.Tensor, activation_fn:str):
    if activation_fn is None:
        return x  # 不使用激活函数
    elif activation_fn == "relu":
        return F.relu(x)
    elif activation_fn == "sigmoid":
        return torch.sigmoid(x)
    elif activation_fn == "tanh":
        return torch.tanh(x)
    elif activation_fn == "leaky_relu":
        return F.leaky_relu(x)
    elif activation_fn == "softmax":
        return F.softmax(x, dim=-1)
    elif activation_fn == "silu" or activation_fn == "swish":
        return F.silu(x)
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")