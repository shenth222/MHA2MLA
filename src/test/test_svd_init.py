import torch
from transformers import AutoModel, AutoTokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

_model_paths=[
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-135M",
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-360M",
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B",
    "/home/binguo/data/models/meta-llama/Llama-3.2-1B",
    "/home/binguo/data/models/meta-llama/Llama-3.2-3B",
    "/home/binguo/data/models/meta-llama/Llama-3.1-8B",
]


model_path = _model_paths[5]
model = AutoModel.from_pretrained(model_path, device_map="auto")

K, V = [], []
for name, module in model.named_modules():
    if "k_proj" in name:
        K.append(module.weight.t())
    if "v_proj" in name:
        V.append(module.weight.t())

print(f"#layer: {len(K)}, shape of W_k: {K[0].shape} W_v: {V[0].shape}")


def loss_func(init, tgt):
    return torch.norm(init - tgt, p="fro").item()


r = 32
err = [0.0] * 6
for k, v in zip(K, V):
    # method I
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down = (U_k[:, :r] + U_v[:, :r]) / 2
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.diag(S_v) @ V_v.t()
    err[0] += loss_func(W_down @ W_up_k, k) + loss_func(W_down @ W_up_v, v)

    # method II
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down_k = U_k
    W_down_v = U_v
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.diag(S_v) @ V_v.t()
    err[1] += loss_func(W_down_k @ W_up_k, k) + loss_func(W_down_v @ W_up_v, v)

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
    err[2] += loss_func(W_down_k @ W_up_k, k) + loss_func(W_down_v @ W_up_v, v)

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
    err[3] += loss_func(W_down @ W_up_k, k) + loss_func(W_down @ W_up_v, v)

    # method V
    U_k, S_k, V_k = torch.svd(k)
    U_k, S_k, V_k = U_k[:, :r], S_k[:r], V_k[:, :r]
    W_down = U_k
    W_down_pseudo_inv = torch.linalg.pinv(W_down)
    W_up_k = torch.diag(S_k) @ V_k.t()
    W_up_v = torch.matmul(W_down_pseudo_inv, v)
    err[4] += loss_func(W_down @ W_up_k, k) + loss_func(W_down @ W_up_v, v)

    # method VI
    U_v, S_v, V_v = torch.svd(v)
    U_v, S_v, V_v = U_v[:, :r], S_v[:r], V_v[:, :r]
    W_down = U_v
    W_down_pseudo_inv = torch.linalg.pinv(W_down)
    W_up_k = torch.matmul(W_down_pseudo_inv, k)
    W_up_v = torch.diag(S_v) @ V_v.t()
    err[5] += loss_func(W_down @ W_up_k, k) + loss_func(W_down @ W_up_v, v)

err_norm = [x / len(K) / len(V) for x in err]
print(err_norm)
