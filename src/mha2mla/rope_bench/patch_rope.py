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

def create_custom_apply_rotary_pos_emb_hf(cfg):

    def apply_rotary_pos_emb_v0(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Full-RoPE
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v1(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

    def apply_rotary_pos_emb_v2(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

    def apply_rotary_pos_emb_v3(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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
            "HIGH-LOW: retain the subspaces with higher 2-norm score"
        )
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        top_k_dim = cfg["top_k_rope_dim"]
        topk_indices = torch.topk(qk_tensor[layer_idx], k=top_k_dim, dim=1)[1]
        mask = torch.zeros_like(qk_tensor[layer_idx])
        mask.scatter_(1, topk_indices, 1)
        mask_for_k = (
            torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(2).to(k.device)
        )
        mask_for_q = torch.repeat_interleave(
            input=mask_for_k, repeats=cfg["n_gqa_group"], dim=1
        ).to(q.device)
        q_embed = torch.where(mask_for_q == 1, q_embed, q)
        k_embed = torch.where(mask_for_k == 1, k_embed, k)
        return q_embed, k_embed

    def apply_rotary_pos_emb_v5(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
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

    version = cfg["partial_rope_version"]
    version_str2int = {
        "full-rope": 0,
        "high": 1,
        "uniform": 2,
        "high-low": 3,
        "2-norm": 4,
        "low": 5,
    }
    if isinstance(version, str):
        version = version_str2int[version]
    if version == 4:
        print("not support now")
        exit()
        with open(cfg["qk_tensor_path"], "rb") as fin:
            qk_tensor = torch.load(fin)
    versions = {
        0: apply_rotary_pos_emb_v0,
        1: apply_rotary_pos_emb_v1,
        2: apply_rotary_pos_emb_v2,
        3: apply_rotary_pos_emb_v3,
        4: apply_rotary_pos_emb_v4,
        5: apply_rotary_pos_emb_v5,
    }
    return versions.get(version, apply_rotary_pos_emb_v0)

def patch_partial_rope(rope_cfg):
    modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(rope_cfg)