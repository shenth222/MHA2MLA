# patch for post init, deleting origin weight, using HDM res
# LlamaModel -> post_init -> LlamaForCasualLM -> post_init
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.cache_utils import Cache
from loguru import logger
import math
import torch
import torch.nn.functional as F

def rebuild_weight(self, num_layers, w1, h1, w2, h2, mode="Q"):
    logger.info(f"Rebuild {mode} weight")
    attn = ["Q", "K", "V", "O"]
    mlp = ["U", "D", "G"]
    if mode in attn:
        for layer_idx in range(num_layers):
            logger.info(f"building layer {layer_idx} {mode} weight")
            commond = f"""
del self.model.layers[layer_idx].self_attn.{mode.lower()}_proj
self.model.layers[layer_idx].self_attn.{mode.lower()}_w1 = w1[layer_idx].cuda()
self.model.layers[layer_idx].self_attn.{mode.lower()}_h1 = h1[layer_idx].cuda()
self.model.layers[layer_idx].self_attn.{mode.lower()}_w2 = w2[layer_idx].cuda()
self.model.layers[layer_idx].self_attn.{mode.lower()}_h2 = h2[layer_idx].cuda()
            """
            exec(commond)

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

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_patch():
    LlamaForCausalLM.rebuild_weight = rebuild_weight
    LlamaAttention.forward = LA_forward