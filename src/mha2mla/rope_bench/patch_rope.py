import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.cache_utils import Cache

from transformers.utils import (
    logging,
)

# logger = logging.get_logger(__name__)

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaAttention
)
from typing import List, Union, Dict, Optional, Tuple

import torch
import math
from torch import nn
from torch.nn import functional as F

from transformers.models.llama.modeling_llama import (
    rotate_half,
    repeat_kv,
    logger,
    StaticCache,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers import Cache
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    logger,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import is_flash_attn_greater_or_equal_2_10

def cal_2_norm(states):
    # bsz, num_heads, q_len, head_dim
    states = torch.norm(
        states.reshape(states.shape[0],states.shape[1],states.shape[2],2,-1).transpose(-1,-2),
        p=2,
        dim=4,
    )
    return states # bsz, num_heads, q_len, head_dim//2

def zero_ablation(q, k):
    # bsz, num_heads, q_len, head_dim
    # baseline = torch.matmul(q, k.transpose(2,3)) # # bsz, num_heads, q_len, q_len
    # contribution = []
    # _, _, slen, head_dim = q.size()
    # for s_idx in range(slen):
    #     q_seq = q[:, :, s_idx, :].unsqueeze(2) # # bsz, num_heads, 1, head_dim
    #     k_seq = k[:, :, s_idx, :].unsqueeze(2)
    #     diff_by_dim = []
    #     for dim_idx in range(head_dim):
    #         q_tmp = q_seq.clone()
    #         k_tmp = k_seq.clone()
    #         q_tmp[:, :, :, dim_idx] = 0
    #         k_tmp[:, :, :, dim_idx] = 0
    #         res_ablation = torch.matmul(q_tmp, k_tmp.transpose(2, 3)) # bsz, num_heads, 1, 1
    #         diff = torch.abs(baseline[:, :, s_idx, s_idx].unsqueeze(-1).unsqueeze(-1) - res_ablation).squeeze(-1).squeeze(-1) # bsz, num_heads
    #         diff_by_dim.append(diff)
    #     diff_by_dim = torch.stack(diff_by_dim, dim=-1) # bsz, num_heads, head_dim
    #     total_diff = torch.sum(diff_by_dim, dim=-1) # bsz, num_heads
    #     contribution_by_seq = diff_by_dim / total_diff.unsqueeze(-1) # bsz, num_heads, head_dim
    #     contribution.append(contribution_by_seq)

    baseline = torch.sum(q * k, dim=-1) # bsz, num_heads, q_len
    contribution = []
    _, _, slen, head_dim = q.size()
    for s_idx in range(slen):
        q_seq = q[:, :, s_idx:s_idx+1, :] # bsz, num_heads, 1, head_dim
        k_seq = k[:, :, s_idx:s_idx+1, :]
        diff_by_dim = []
        for dim_idx in range(head_dim):
            q_tmp = q_seq.clone()
            k_tmp = k_seq.clone()
            q_tmp[:, :, :, dim_idx] = 0
            k_tmp[:, :, :, dim_idx] = 0
            res_ablation = torch.matmul(q_tmp, k_tmp.transpose(2, 3)).squeeze() # bsz, num_heads # sum(q*k)
            diff = torch.abs(baseline[:, :, s_idx] - res_ablation) # bsz, num_heads
            diff_by_dim.append(diff)
        diff_by_dim = torch.stack(diff_by_dim, dim=-1) # bsz, num_heads, head_dim
        total_diff = torch.sum(diff_by_dim, dim=-1) # bsz, num_heads
        contribution_by_seq = diff_by_dim / total_diff.unsqueeze(-1) # bsz, num_heads, head_dim
        contribution.append(contribution_by_seq)
    return torch.stack(contribution, dim=2) # bsz, num_heads, seq_len, head_dim

def find_indices(contribution, threshold, max_dim):
    # 假设 contribution 的形状为 (bsz, num_kv_heads, seq_len, head_dim)

    # 1. 对最后一维进行降序排序
    sorted_values, sorted_indices = torch.sort(contribution, dim=-1, descending=True)

    # 2. 计算累加值
    cumulative_sum = torch.cumsum(sorted_values, dim=-1)

    # 3. 找到满足累加值 >= threshold 的最小索引
    threshold_mask = cumulative_sum >= threshold
    valid_indices = torch.argmax(threshold_mask.int(), dim=-1)  # 找到第一个满足条件的索引

    # 4. 如果无法达到 threshold，则选择 max_dim 个最大的索引
    num_elements = torch.minimum(valid_indices + 1, torch.tensor(max_dim, device=contribution.device))

    # 5. 构造最终的索引掩码
    # final_mask = torch.zeros_like(contribution, dtype=torch.bool)
    # for i in range(contribution.size(0)):  # 遍历 batch
    #     for j in range(contribution.size(1)):  # 遍历 num_kv_heads
    #         for k in range(contribution.size(2)):  # 遍历 seq_len
    #             final_mask[i, j, k, sorted_indices[i, j, k, :num_elements[i, j, k]]] = True

    _, _, _, head_dim = contribution.shape
    arange = torch.arange(head_dim, device=contribution.device).view(1, 1, 1, -1)  # 形状为 (1, 1, 1, head_dim)
    num_elements_expanded = num_elements.unsqueeze(-1)  # 形状为 (bsz, num_kv_heads, seq_len, 1)
    mask = arange < num_elements_expanded  # 形状为 (bsz, num_kv_heads, seq_len, head_dim)

    # 根据排序索引还原掩码
    final_mask = torch.zeros_like(contribution, dtype=torch.bool)
    final_mask.scatter_(-1, sorted_indices, mask)

    return final_mask

def create_custom_apply_rotary_pos_emb_hf(cfg):

    def apply_rotary_pos_emb_v0(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # Full-RoPE
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v1(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # retain the fastest-rotating (high-frequency) subspaces
        logger.warning_once(
            "HIGH: retain the fastest-rotating (high-frequency) subspaces"
        )
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        keep_dim = cfg["top_k_rope_dim"]
        if keep_dim <= 0:
            return q, k
        elif keep_dim >= q.size(-1):
            return q_embed, k_embed
        half = q.size(-1) // 2
        q_embed = torch.cat(
            (
                q_embed[..., :keep_dim],
                q[..., keep_dim:half],
                q_embed[..., half : half + keep_dim],
                q[..., half + keep_dim :],
            ),
            -1,
        )
        k_embed = torch.cat(
            (
                k_embed[..., :keep_dim],
                k[..., keep_dim:half],
                k_embed[..., half : half + keep_dim],
                k[..., half + keep_dim :],
            ),
            -1,
        )
        return q_embed, k_embed

    def apply_rotary_pos_emb_v2(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # select subspaces with equidistant intervals
        logger.warning_once(
            "UNIFORM: select subspaces with equidistant intervals"
        )
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        indices = torch.arange(
            cfg["uniform_start_point"], q.size(-1), cfg["uniform_step"], device=q.device
        )
        q[..., indices] = q_embed[..., indices]
        k[..., indices] = k_embed[..., indices]
        return q, k

    def apply_rotary_pos_emb_v3(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # retain the fastest-rotating (high-frequency) subspaces and the slowest-rotating (low-frequency) subspaces
        logger.warning_once(
            "HIGH-LOW: retain the fastest-rotating (high-frequency) subspaces and the slowest-rotating (low-frequency) subspaces"
        )
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        top_k_dim, last_k_dim = cfg["top_k_rope_dim"], cfg["last_k_rope_dim"]
        # assert top_k_dim + last_k_dim <= q.size(-1)
        half = q.size(-1) // 2
        qs = [
            q_embed[..., :top_k_dim],
            q[..., top_k_dim : half - last_k_dim],
            q_embed[..., half - last_k_dim : half + top_k_dim],
            q[..., half + top_k_dim : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k_embed[..., :top_k_dim],
            k[..., top_k_dim : half - last_k_dim],
            k_embed[..., half - last_k_dim : half + top_k_dim],
            k[..., half + top_k_dim : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v4(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # retain the subspaces with higher 2-norm score
        # a strong hypothesis is that a unimportant component of key_states is also unimportant in the future
        logger.warning_once(
            "2-Norm: retain the subspaces with higher 2-norm score"
        )
        cos = cos.unsqueeze(unsqueeze_dim) # bsz, 1, q_len, head_dim
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin) # bsz, num_heads, q_len, head_dim
        k_embed = (k * cos) + (rotate_half(k) * sin) # bsz, num_kv_heads, q_len, head_dim

        repeat_k = repeat_kv(k, cfg["n_gqa_group"]) # bsz, num_heads, q_len, head_dim
        q_norm = cal_2_norm(q).mean(dim=2) # bsz, num_heads, head_dim//2
        k_norm = cal_2_norm(repeat_k).mean(dim=2) # bsz, num_heads, head_dim//2
        qk_norm = q_norm * k_norm # bsz, num_heads, head_dim//2

        if cfg["n_gqa_group"] > 1:
            qk_norm = qk_norm.reshape(
                qk_norm.size(0),
                qk_norm.size(1) // cfg["n_gqa_group"],
                cfg["n_gqa_group"],
                qk_norm.size(2)
            )
            qk_norm = qk_norm.mean(dim=2) # bsz, num_kv_heads, head_dim//2

        top_k_dim = cfg["top_k_rope_dim"]
        _, topk_indices = torch.topk(qk_norm, k=top_k_dim, dim=-1)
        mask = torch.zeros_like(qk_norm, dtype=torch.float32)
        mask.scatter_(-1, topk_indices, 1) # bsz, num_kv_heads, head_dim//2
        mask_for_k = (torch.cat((mask, mask), dim=-1).unsqueeze(2)) # bsz, num_kv_heads, 1, head_dim
        mask_for_q = torch.repeat_interleave(input=mask_for_k, repeats=cfg["n_gqa_group"], dim=1)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v5(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        # retain the slowest-rotating (low-frequency) subspaces
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        last_k_dim = cfg["last_k_rope_dim"]
        half = q.size(-1) // 2
        qs = [
            q[..., : half - last_k_dim],
            q_embed[..., half - last_k_dim : half],
            q[..., half : half + half - last_k_dim],
            q_embed[..., half + half - last_k_dim :],
        ]
        ks = [
            k[..., : half - last_k_dim],
            k_embed[..., half - last_k_dim : half],
            k[..., half : half + half - last_k_dim],
            k_embed[..., half + half - last_k_dim :],
        ]
        q_embed = torch.cat([q for q in qs if q != []], -1)
        k_embed = torch.cat([k for k in ks if k != []], -1)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v6(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        logger.warning_once(
            "Contribution: retain the subspaces with higher contribution"
        )
        from IPython import embed
        if layer_idx == 0:
            embed()
        cos = cos.unsqueeze(unsqueeze_dim) #1, 1, q_len, head_dim
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin) # bsz, num_heads, q_len, head_dim
        k_embed = (k * cos) + (rotate_half(k) * sin) # bsz, num_kv_heads, q_len, head_dim

        repeat_k = repeat_kv(k, cfg["n_gqa_group"]) # bsz, num_heads, q_len, head_dim

        contribution = zero_ablation(q, repeat_k) # bsz, num_heads, seq_len, head_dim
        if cfg["n_gqa_group"] > 1:
            contribution = contribution.reshape(
                contribution.size(0),
                contribution.size(1) // cfg["n_gqa_group"],
                cfg["n_gqa_group"],
                contribution.size(2),
                contribution.size(3)
            ).mean(dim=2) # bsz, num_kv_heads, seq_len, head_dim

        top_k_dim = cfg["top_k_rope_dim"]
        _, topk_indices = torch.topk(contribution, k=top_k_dim, dim=-1)
        mask = torch.zeros_like(contribution, dtype=torch.float32)
        mask_for_k = mask.scatter_(-1, topk_indices, 1) # bsz, num_kv_heads, seq_len, head_dim
        mask_for_q = torch.repeat_interleave(input=mask_for_k, repeats=cfg["n_gqa_group"], dim=1)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        if layer_idx == 0 and cos[0][0][0][0] != 1:
            exit()
        return q_embed, k_embed
    
    def apply_rotary_pos_emb_v7(q, k, cos, sin, position_ids=None, layer_idx=0, unsqueeze_dim=1):
        logger.warning_once(
            "Recovery: retain the subspaces that reaches recovery rate threshold \n" \
            f"Max component: {cfg['max_component']}"
        )
        cos = cos.unsqueeze(unsqueeze_dim) # bsz, 1, q_len, head_dim
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin) # bsz, num_heads, q_len, head_dim
        k_embed = (k * cos) + (rotate_half(k) * sin) # bsz, num_kv_heads, q_len, head_dim

        repeat_k = repeat_kv(k, cfg["n_gqa_group"]) # bsz, num_heads, q_len, head_dim

        contribution = zero_ablation(q, repeat_k) # bsz, num_heads, seq_len, head_dim
        if cfg["n_gqa_group"] > 1:
            contribution = contribution.reshape(
                contribution.size(0),
                contribution.size(1) // cfg["n_gqa_group"],
                cfg["n_gqa_group"],
                contribution.size(2),
                contribution.size(3)
            ).mean(dim=2) # bsz, num_kv_heads, seq_len, head_dim

        contribution = torch.nn.functional.softmax(contribution, dim=-1)

        threshold = cfg["recovery_rate"]
        max_dim = int(math.ceil(k.shape[-1] * 0.2)) if cfg["max_component"] else k.shape[-1]

        mask_for_k = find_indices(contribution, threshold, max_dim)
        mask_for_q = torch.repeat_interleave(input=mask_for_k, repeats=cfg["n_gqa_group"], dim=1)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    version = cfg["partial_rope_version"]
    version_str2int = {
        "full-rope": 0,
        "high": 1,
        "uniform": 2,
        "high-low": 3,
        "2-norm": 4,
        "low": 5,
        "contribution": 6,
        "accumulate": 7,
    }
    if isinstance(version, str):
        version = version_str2int[version]
    # if version == 4:
    #     print("not support now")
    #     exit()
    #     with open(cfg["qk_tensor_path"], "rb") as fin:
    #         qk_tensor = torch.load(fin)
    versions = {
        0: apply_rotary_pos_emb_v0,
        1: apply_rotary_pos_emb_v1,
        2: apply_rotary_pos_emb_v2,
        3: apply_rotary_pos_emb_v3,
        4: apply_rotary_pos_emb_v4,
        5: apply_rotary_pos_emb_v5,
        6: apply_rotary_pos_emb_v6,
        7: apply_rotary_pos_emb_v7,
    }
    return versions.get(version, apply_rotary_pos_emb_v0)

def attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, layer_idx=self.layer_idx)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def patch_partial_rope(rope_cfg):
    modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(rope_cfg)
    modeling_llama.LlamaAttention.forward = attn_forward
