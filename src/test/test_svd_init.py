import torch
from transformers import AutoModel, AutoTokenizer
import os
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

_model_paths=[
    "~/data/models/HuggingFaceTB/SmolLM-135M",
    "~/data/models/HuggingFaceTB/SmolLM-360M",
    "~/data/models/HuggingFaceTB/SmolLM-1.7B",
    "~/data/models/meta-llama/Llama-3.2-1B",
    "~/data/models/meta-llama/Llama-3.2-3B",
    "~/data/models/meta-llama/Llama-3.1-8B",
]


model_path = _model_paths[5]
r = 32

# 保存到同目录下的log文件
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='a')
logger = logging.getLogger(__name__)
model = AutoModel.from_pretrained(model_path, device_map="auto")

K, V = [], []
K_nums, V_nums = 0, 0
for name, module in model.named_modules():
    if "k_proj" in name:
        K.append(module.weight.t())
        K_nums += module.weight.numel()
    if "v_proj" in name:
        V.append(module.weight.t())
        V_nums += module.weight.numel()

logging.info(f"\n\n\n")
logging.info(f"model_path: {model_path}")
logging.info(f"r: {r}")
print(f"#layer: {len(K)}, shape of W_k: {K[0].shape} W_v: {V[0].shape}")
print(f"Total number of parameters in W_k: {K_nums}, W_v: {V_nums}")
logging.info(f"#layer: {len(K)}, shape of W_k: {K[0].shape} W_v: {V[0].shape}")
logging.info(f"Total number of parameters in W_k: {K_nums}, W_v: {V_nums}")


def loss_func(init, tgt):
    return torch.norm(init - tgt, p="fro").item()


err_K = [0.0] * 6
err_V = [0.0] * 6

for k, v in zip(K, V):
    # method I
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down = (U_k[:, :r] + U_v[:, :r]) / 2
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.diag(S_v) @ V_v.t()
    err_K[0] += loss_func(W_down @ W_up_k, k)
    err_V[0] += loss_func(W_down @ W_up_v, v)

    # method II
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down_k = U_k
    W_down_v = U_v
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.diag(S_v) @ V_v.t()
    err_K[1] += loss_func(W_down_k @ W_up_k, k)
    err_V[1] += loss_func(W_down_v @ W_up_v, v)

    # method III
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    Sigma_k_half = torch.diag(torch.sqrt(S_k))
    Sigma_v_half = torch.diag(torch.sqrt(S_v))
    W_down_k = U_k @ Sigma_k_half
    W_down_v = U_v @ Sigma_v_half
    W_up_k = Sigma_k_half @ V_k.t()
    W_up_v = Sigma_v_half @ V_v.t()
    err_K[2] += loss_func(W_down_k @ W_up_k, k)
    err_V[2] += loss_func(W_down_v @ W_up_v, v)

    # method IV
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    Sigma_k_half = torch.diag(torch.sqrt(S_k))
    Sigma_v_half = torch.diag(torch.sqrt(S_v))
    W_down_k = U_k @ Sigma_k_half
    W_down_v = U_v @ Sigma_v_half
    W_down = (W_down_k + W_down_v) / 2
    W_up_k = Sigma_k_half @ V_k.t()
    W_up_v = Sigma_v_half @ V_v.t()
    err_K[3] += loss_func(W_down @ W_up_k, k)
    err_V[3] += loss_func(W_down @ W_up_v, v)

    # method V
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    W_down = U_k
    W_down_pseudo_inv = torch.linalg.pinv(W_down)
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.matmul(W_down_pseudo_inv, v)
    err_K[4] += loss_func(W_down @ W_up_k, k)
    err_V[4] += loss_func(W_down @ W_up_v, v)

    # method VI
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down = U_v
    W_down_pseudo_inv = torch.linalg.pinv(W_down)
    W_up_k = torch.matmul(W_down_pseudo_inv, k)
    W_up_v = torch.diag(S_v) @ V_v.t()
    err_K[5] += loss_func(W_down @ W_up_k, k)
    err_V[5] += loss_func(W_down @ W_up_v, v)

    # method VII
    U_kv,S_kv,V_kv = torch.svd(torch.cat([k,v],dim=0))
    U_kv,S_kv,V_kv = U_kv[:,:r],S_kv[:r],V_kv[:,:r]
    W_down = U_kv
    split_size = V_kv.shape[0]//2
    W_up_k = torch.split(V_kv,split_size,dim=0)[0].t()
    W_up_v = torch.split(V_kv,split_size,dim=0)[1].t()
    W_up_k = torch.diag(S_kv) @ W_up_k
    W_up_v = torch.diag(S_kv) @ W_up_v
    err_K[6] += loss_func(W_down @ W_up_k, k)
    err_V[6] += loss_func(W_down @ W_up_v, v)


err_norm_K = [x / K_nums for x in err_K]
err_norm_V = [x / V_nums for x in err_V]

# 使用科学计数法并保留三位小数格式化输出
def format_scientific(values):
    return [f"{v:.3e}" for v in values]

err_norm_K_formatted = format_scientific(err_norm_K)
err_norm_V_formatted = format_scientific(err_norm_V)

print(f"err_norm_K: {err_norm_K_formatted}")
print(f"err_norm_V: {err_norm_V_formatted}")
logging.info(f"err_norm_K: {err_norm_K_formatted}")
logging.info(f"err_norm_V: {err_norm_V_formatted}")

err_norm = [(x + y)/(K_nums+V_nums) for x, y in zip(err_K, err_V)]
err_norm_formatted = format_scientific(err_norm)

print(f"err_norm: {err_norm_formatted}")
logging.info(f"err_norm: {err_norm_formatted}")