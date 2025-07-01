# patch for post init, deleting origin weight, using HDM res
# LlamaModel -> post_init -> LlamaForCasualLM -> post_init
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv, LlamaMLP
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.cache_utils import Cache
from loguru import logger
import math
import torch
import torch.nn.functional as F
import sys
sys.path.append("/data/shenth/work/MHA2MLA/src")
from HDM.utils import load_HDM_res

def mean_merge(input_layers: list):
    tensor = sum(input_layers)
    return tensor / len(input_layers)

def svd_merge(input_layers: list):
    rank = 64
    row, col = input_layers[0].shape
    tensor = torch.cat(input_layers, dim=1)
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]
    kernel = U @ torch.diag_embed(S)
    v = torch.split(Vt, col, dim=1)
    return 

def merge_layer(self, strategy, mode, factor, merge_impl="mean", rank=512, dtype=torch.float16):
    logger.info(f"Merge {mode} weight")
    factors = ["h1", "h2"]
    if factor not in factors: logger.error(f"No *_{factor} weight")
    attn = ["Q", "K", "V", "O"]
    mlp = ["U", "D", "G"]
    mode_map = {
        "Q": "q",
        "K": "k",
        "V": "v",
        "O": "o",
        "U": "up",
        "D": "down",
        "G": "gate"
    }
    if mode not in attn and mode not in mlp: logger.error(f"No {mode}_* weight")
    module = "self_attn" if mode in attn else "mlp"
    impl = ["mean", "svd"]
    if merge_impl not in impl: logger.error(f"No {merge_impl} impl for merge")
    if type(strategy) is int:
        strategy = list(range(strategy))
    _, h1, _, h2 = load_HDM_res("../HDM_res", mode=mode, rank=rank)
    commond = f"""
tensor = {merge_impl}_merge({factor}).to(dtype).cuda()
    """
    exec(commond)
    for layer_idx in strategy:
        logger.info(f"Merge layer {layer_idx} {module} {mode} weight")
        commond = f"""
del self.model.layers[layer_idx].{module}.{mode_map[mode]}_{factor}
torch.cuda.empty_cache()
self.model.layers[layer_idx].{module}.{mode_map[mode]}_{factor} = tensor
        """
        exec(commond)


def rebuild_weight(self, strategy, mode="Q", rank=512, dtype=torch.float16):
    logger.info(f"Rebuild {mode} weight")
    w1, h1, w2, h2 = load_HDM_res("../HDM_res", mode=mode, rank=rank)
    assert w1 is not None, f"Fail to load HDM weights"
    attn = ["Q", "K", "V", "O"]
    mlp = ["U", "D", "G"]
    mode_map = {
        "Q": "q",
        "K": "k",
        "V": "v",
        "O": "o",
        "U": "up",
        "D": "down",
        "G": "gate"
    }
    if type(strategy) is int:
        strategy = list(range(strategy))
    if mode in attn:
        for layer_idx in strategy:
            logger.info(f"building layer {layer_idx} attn {mode} weight")
            commond = f"""
del self.model.layers[layer_idx].self_attn.{mode_map[mode]}_proj
torch.cuda.empty_cache()
self.model.layers[layer_idx].self_attn.{mode_map[mode]}_w1 = w1[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].self_attn.{mode_map[mode]}_h1 = h1[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].self_attn.{mode_map[mode]}_w2 = w2[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].self_attn.{mode_map[mode]}_h2 = h2[layer_idx].to(dtype).cuda()
            """
            exec(commond)
    elif mode in mlp:
        for layer_idx in strategy:
            logger.info(f"building layer {layer_idx} mlp {mode} weight")
            commond = f"""
del self.model.layers[layer_idx].mlp.{mode_map[mode]}_proj
torch.cuda.empty_cache()
self.model.layers[layer_idx].mlp.{mode_map[mode]}_w1 = w1[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].mlp.{mode_map[mode]}_h1 = h1[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].mlp.{mode_map[mode]}_w2 = w2[layer_idx].to(dtype).cuda()
self.model.layers[layer_idx].mlp.{mode_map[mode]}_h2 = h2[layer_idx].to(dtype).cuda()
            """
            exec(commond)
    torch.cuda.empty_cache()

def LA_forward(
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
    bsz, q_len, _ = hidden_states.size()

    if hasattr(self, 'q_proj'):
        query_states = self.q_proj(hidden_states)
    else:
        query_states = hidden_states @ ((self.q_w1 @ self.q_h1) * (self.q_w2 @ self.q_h2)).T

    if hasattr(self, 'k_proj'):
        key_states = self.k_proj(hidden_states)
    else:
        key_states = hidden_states @ ((self.k_w1 @ self.k_h1) * (self.k_w2 @ self.k_h2)).T

    if hasattr(self, 'v_proj'):
        value_states = self.v_proj(hidden_states)
    else:
        value_states = hidden_states @ ((self.v_w1 @ self.v_h1) * (self.v_w2 @ self.v_h2)).T
    
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
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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

    if hasattr(self, 'o_proj'):
        attn_output = self.o_proj(attn_output)
    else:
        attn_output = attn_output @ ((self.o_w1 @ self.o_h1) * (self.o_w2 @ self.o_h2)).T
    

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def LMLP_forward(self, x):
    # from IPython import embed
    # embed()
    # exit()
    up = self.up_proj(x) if hasattr(self, 'up_proj') else x @ ((self.up_w1 @ self.up_h1) * (self.up_w2 @ self.up_h2))
    gate = self.gate_proj(x) if hasattr(self, 'gate_proj') else x @ ((self.gate_w1 @ self.gate_h1) * (self.gate_w2 @ self.gate_h2))
    down_proj = self.down_proj(self.act_fn(gate * up)) if hasattr(self, 'down_proj') else self.act_fn(gate * up) @ ((self.down_w1 @ self.down_h1) * (self.down_w2 @ self.down_h2)).T
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    del up, gate
    torch.cuda.empty_cache()
    return down_proj

def enable_llama_patch():
    LlamaForCausalLM.rebuild_weight = rebuild_weight
    LlamaAttention.forward = LA_forward
    LlamaMLP.forward = LMLP_forward
    LlamaForCausalLM.merge_layer = merge_layer