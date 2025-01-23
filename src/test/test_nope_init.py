import torch
from transformers import AutoModel, AutoTokenizer
import os
import copy
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

_model_paths = [
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-135M",
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-360M",
    "/home/binguo/data/models/HuggingFaceTB/SmolLM-1.7B",
    "/home/binguo/data/models/meta-llama/Llama-3.2-1B",
    "/home/binguo/data/models/meta-llama/Llama-3.2-3B",
    "/home/binguo/data/models/meta-llama/Llama-3.1-8B",
]


model_path = _model_paths[0]


# 保存到同目录下的log文件
logging.basicConfig(level=logging.DEBUG, filename="log.txt", filemode="a")
logger = logging.getLogger(__name__)
model = AutoModel.from_pretrained(model_path, device_map="auto")


def get_rope_config(model, rope_scale=8):
    assert rope_scale in [4, 8]

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    head_num = model.config.num_key_value_heads

    kwargs = {
        "head_dim": head_dim,
        "head_num": head_num,
    }

    rope_cfgs = [
        {
            "partial_rope_version": 1,  # 1 2 3 4 5
            "top_k_rope_dim": head_dim//(2*rope_scale),
            "last_k_rope_dim": 0,
            "uniform_start_point": 0,
            "uniform_step": rope_scale,
            "qk_tensor_path": "/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth",
            "n_gqa_group": 3,
        },
        {
            "partial_rope_version": 2,  # 1 2 3 4 5
            "top_k_rope_dim": 0,
            "last_k_rope_dim": 0,
            "uniform_start_point": 0,
            "uniform_step": rope_scale,
            "qk_tensor_path": "/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth",
            "n_gqa_group": 3,
        },
        {
            "partial_rope_version": 3,  # 1 2 3 4 5
            "top_k_rope_dim": head_dim//(4*rope_scale),
            "last_k_rope_dim": head_dim//(4*rope_scale),
            "uniform_start_point": 0,
            "uniform_step": rope_scale,
            "qk_tensor_path": "/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth",
            "n_gqa_group": 3,
        },
        {
            "partial_rope_version": 4,  # 1 2 3 4 5
            "top_k_rope_dim": head_dim//(2*rope_scale),
            "last_k_rope_dim": 0,
            "uniform_start_point": 0,
            "uniform_step": rope_scale,
            "qk_tensor_path": "/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth",
            "n_gqa_group": 3,
        },
        {
            "partial_rope_version": 5,  # 1 2 3 4 5
            "top_k_rope_dim": 0,
            "last_k_rope_dim": head_dim//(2*rope_scale),
            "uniform_start_point": 0,
            "uniform_step": rope_scale,
            "qk_tensor_path": "/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth",
            "n_gqa_group": 3,
        },
    ]
    return kwargs, rope_cfgs


class IndexForNope:
    @staticmethod
    def get_index_for_nope_v0(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        return torch.empty((head_dim))

    @staticmethod
    def get_index_for_nope_v1(rope_cfg, **kwargs):
        keep_dim = rope_cfg["top_k_rope_dim"]
        head_dim = kwargs["head_dim"]
        if keep_dim <= 0:
            return torch.arange(head_dim)
        elif keep_dim >= head_dim:
            return torch.empty((head_dim))
        else:
            half = head_dim // 2
            return torch.cat(
                [
                    torch.arange(start=keep_dim, end=half, step=1),
                    torch.arange(start=half + keep_dim, end=half + half, step=1),
                ],
                dim=0,
            )

    @staticmethod
    def get_index_for_nope_v2(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        indices = torch.arange(head_dim)
        indices_to_remove = torch.arange(
            rope_cfg["uniform_start_point"], head_dim, rope_cfg["uniform_step"]
        )
        mask = torch.ones(head_dim, dtype=torch.bool)
        mask[indices_to_remove] = False
        return indices[mask]

    @staticmethod
    def get_index_for_nope_v3(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        top_k_dim, last_k_dim = rope_cfg["top_k_rope_dim"], rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        assert top_k_dim + last_k_dim <= half
        return torch.cat(
            [
                torch.arange(start=top_k_dim, end=half - last_k_dim, step=1),
                torch.arange(
                    start=half + top_k_dim, end=half + half - last_k_dim, step=1
                ),
            ],
            dim=0,
        )

    @staticmethod
    def get_index_for_nope_v4(rope_cfg, **kwargs):
        with open(rope_cfg["qk_tensor_path"], "rb") as fin:
            qk_tensor = torch.load(fin).cuda()  # [layer_num, k_head_num, head_dim//2]
            assert len(qk_tensor.size()) == 3
        layer_id = kwargs["layer_id"]
        top_k_dim = rope_cfg["top_k_rope_dim"]
        head_dim = kwargs["head_dim"]
        topk_indices = torch.topk(qk_tensor[layer_id], k=top_k_dim, dim=1)[1]
        all_indices = (
            torch.arange(qk_tensor[layer_id].size(1), device=qk_tensor.device)
            .unsqueeze(0)
            .repeat(qk_tensor[layer_id].size(0), 1)
        )
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(1, topk_indices, False)
        remaining_indices_for_k = all_indices[mask].view(all_indices.size(0), -1)
        remaining_indices_for_k = torch.cat([remaining_indices_for_k,remaining_indices_for_k+head_dim//2],dim=-1)
        return remaining_indices_for_k

    @staticmethod
    def get_index_for_nope_v5(rope_cfg, **kwargs):
        head_dim = kwargs["head_dim"]
        last_k_rope_dim = rope_cfg["last_k_rope_dim"]
        half = head_dim // 2
        return torch.cat(
            [
                torch.arange(start=0, end=half - last_k_rope_dim, step=1),
                torch.arange(start=half, end=half + half - last_k_rope_dim, step=1),
            ],
            dim=0,
        )

    @staticmethod
    def get_index_for_nope(rope_cfg, **kwargs):
        version = rope_cfg["partial_rope_version"]
        versions = {
            0: IndexForNope.get_index_for_nope_v0,
            1: IndexForNope.get_index_for_nope_v1,
            2: IndexForNope.get_index_for_nope_v2,
            3: IndexForNope.get_index_for_nope_v3,
            4: IndexForNope.get_index_for_nope_v4,
            5: IndexForNope.get_index_for_nope_v5,
        }
        index_func = versions[version]
        remaining_indices_for_k = index_func(rope_cfg, **kwargs)
        add_values = torch.arange(
            start=0,
            end=kwargs["head_num"] * kwargs["head_dim"],
            step=kwargs["head_dim"],
        )
        if version == 4:
            remaining_indices_for_k = (
                remaining_indices_for_k.cpu() + add_values.unsqueeze(1)
            )
        else:
            remaining_indices_for_k = remaining_indices_for_k.unsqueeze(
                0
            ) + add_values.unsqueeze(1)
        remaining_indices_for_k = remaining_indices_for_k.view(-1)
        return remaining_indices_for_k


def cal_svd_init_err(model,r,rope_cfg,kwargs:dict):

    K, V = [], []
    K_nums, V_nums = 0, 0
    for name, module in model.named_modules():
        if name.startswith("layer") and len(name.split(".")) > 1:
            layer_id = int(name.split(".")[1])
        if "k_proj" in name:
            W_k = module.weight.t()
            remaining_indices_for_k = IndexForNope.get_index_for_nope(
                rope_cfg, layer_id=layer_id, **kwargs
            )
            W_k = W_k[..., remaining_indices_for_k]
            K.append(W_k)
            K_nums += W_k.numel()
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
    logging.info(f"rope_cfg: {rope_cfg}")
    logging.info(f"kwargs: {kwargs}")

    def loss_func(init, tgt):
        return torch.norm(init - tgt, p="fro").item()

    err_K = [0.0] * 7
    err_V = [0.0] * 7

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
        U_kv, S_kv, V_kv = torch.svd(torch.cat([k, v], dim=1))
        U_kv, S_kv, V_kv = U_kv[:, :r], S_kv[:r], V_kv[:, :r]
        W_down = U_kv
        split_sizes = [k.size(1), v.size(1)]
        W_up_k, W_up_v = torch.split(V_kv, split_sizes, dim=0)
        W_up_k = torch.diag(S_kv) @ W_up_k.t()
        W_up_v = torch.diag(S_kv) @ W_up_v.t()
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

    err_norm = [(x + y) / (K_nums + V_nums) for x, y in zip(err_K, err_V)]
    err_norm_formatted = format_scientific(err_norm)

    print(f"err_norm: {err_norm_formatted}")
    logging.info(f"err_norm: {err_norm_formatted}")
    return {
        "err_norm_K": err_norm_K_formatted,
        "err_norm_V": err_norm_V_formatted,
        "err_norm": err_norm_formatted,
    }


def lists_to_md_table(*lists):
    """
    将多个列表转换为 Markdown 表格，每个列表作为一行。
    :param lists: 多个列表，例如 [1, 2, 3], ['a', 'b', 'c']
    :return: Markdown 表格的字符串
    """
    # 检查所有列表长度是否一致
    if len(set(len(lst) for lst in lists)) > 1:
        raise ValueError("所有列表的长度必须一致")
    # 生成表头分隔线
    header_separator = "| " + " | ".join(["---"] * len(lists[0])) + " |"
    # 生成表格内容
    table_rows = []
    for lst in lists:
        row = "| " + " | ".join(map(str, lst)) + " |"
        table_rows.append(row)
    # 组合成完整的表格
    md_table = "\n".join(table_rows[0:1]+[header_separator] + table_rows[1:])
    return md_table

if __name__ == "__main__":
    outputs=[]
    outputs.append(
        ["r","rope_scale"]+["Method I"]*3+["Method II"]*3+["Method III"]*3+["Method IV"]*3+["Method V"]*3+["Method VI"]*3+["Method VII"]*3
    )
    outputs.append(
        ["r","rope_scale"]+["K","V","KV"]*7
    )
    for r in [8,16]:
        r = r*3
        for rope_scale in [4,8]:
            model=AutoModel.from_pretrained(model_path, device_map="auto")
            kwargs, rope_cfgs = get_rope_config(model, rope_scale=rope_scale)
            for rope_cfg in rope_cfgs:
                output=cal_svd_init_err(model,r,rope_cfg,kwargs)
                output=[x for subset in zip(output["err_norm_K"],output["err_norm_V"],output["err_norm"]) for x in subset]
                output=[r,rope_scale]+output
                outputs.append(output)
    with open("output.md", "w") as f:
        f.write(lists_to_md_table(*outputs))
