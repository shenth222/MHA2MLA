from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from math import ceil

import math
import transformers.modeling_attn_mask_utils
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    logger,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from transformers.models.llama import modeling_llama
from transformers.cache_utils import Cache
from .NopeIndex import IndexForNope


class CustomLlamaMLAForInfer(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.nope_mask = IndexForNope.get_index_for_nope(
            config.RoPE,
            head_dim=self.head_dim,
            head_num=self.num_key_value_heads,
            layer_idx=layer_idx,
        )
        self.is_share_W_down = bool(config.SVD["method"] not in [2, 3])
        self.low_rank = config.SVD["low_rank"]

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )

        self.W_k_r = nn.Linear(
            self.hidden_size,
            (self.nope_mask == False).sum().item(),
            bias=config.attention_bias,
        )
        self.W_down_k = nn.Linear(
            self.hidden_size,
            self.low_rank * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        if not self.is_share_W_down:
            self.W_down_v = nn.Linear(
                self.hidden_size,
                self.low_rank * self.num_key_value_heads,
                bias=config.attention_bias,
            )
        self.W_up_k = nn.Linear(
            self.low_rank * self.num_key_value_heads,
            self.num_key_value_heads * self.head_dim
            - (self.nope_mask == False).sum().item(),
            bias=config.attention_bias,
        )
        self.W_up_v = nn.Linear(
            self.low_rank * self.num_key_value_heads,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.is_inference = False

    def matrix_fusion(self):
        assert self.config.SVD["method"] == 7, "only support SVD method 7 which is SVD_{joint}"
        self.is_inference = True
        # q_proj
        group_size = self.num_heads // self.num_key_value_heads
        nope_mask_for_q = self.nope_mask.reshape(self.num_key_value_heads,1,self.head_dim).repeat_interleave(group_size, dim=-2).reshape(-1)
        self.nope_mask_for_q = nope_mask_for_q
        self.W_q_r = nn.Linear(
            in_features=self.hidden_size,
            out_features=(~nope_mask_for_q).sum().item(),
            dtype=self.q_proj.weight.dtype,
            device=self.q_proj.weight.device,
            bias=False,
        )
        self.W_q_r.weight[:] = self.q_proj.weight.T[..., ~self.nope_mask_for_q].T
        self.W_down_q = nn.Linear(
            in_features=self.hidden_size,
            out_features= self.num_heads * self.num_key_value_heads * self.low_rank,
            dtype=self.q_proj.weight.dtype,
            device=self.q_proj.weight.device,
            bias=False,
        )
        # GQA
        W_q_nope = []
        W_o_proj = []
        for i in range(self.num_key_value_heads):
            # q
            head_nope_mask = self.nope_mask[...,i* self.head_dim:(i+1)*self.head_dim]
            head_k_nope_dim = self.W_up_k.out_features // self.num_key_value_heads
            head_W_up_k = self.W_up_k.weight[i*head_k_nope_dim:(i+1)*head_k_nope_dim,:]
            # o
            head_W_up_v = self.W_up_v.weight[i*self.head_dim:(i+1)*self.head_dim,:]
            for j in range(group_size):
                # q
                head_w_q_nope = self.q_proj.weight.T[..., (i*group_size+j)*self.head_dim:(i*group_size+j+1)*self.head_dim]
                head_w_q_nope = head_w_q_nope[...,head_nope_mask]
                W_q_nope.append(head_w_q_nope@(head_W_up_k))
                # o
                head_w_o_proj = self.o_proj.weight[...,(i*group_size+j)*self.head_dim:(i*group_size+j+1)*self.head_dim]
                W_o_proj.append((head_W_up_v.T)@(head_w_o_proj.T))
        W_q_nope = torch.cat(W_q_nope,dim=-1)
        o_oroj_weight = torch.cat(W_o_proj,dim=0)
        self.W_down_q.weight[:] = W_q_nope.T
        self.o_proj = nn.Linear(
            in_features=self.num_heads * self.num_key_value_heads * self.low_rank ,
            out_features=self.hidden_size,
            dtype=self.q_proj.weight.dtype,
            device=self.q_proj.weight.device,
            bias=False,
        )
        self.o_proj.weight[:] = o_oroj_weight.T
        self.q_proj = nn.Sequential()


    def get_qk_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        assert not self.config.pretraining_tp > 1, "not support pretraining_tp>1"
        # prepare q_r and k_r
        query_states = torch.zeros(
            (bsz,q_len,self.num_heads * self.head_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        key_states = torch.zeros(
            (bsz, q_len, self.num_key_value_heads * self.head_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        q_r = self.W_q_r(hidden_states)
        k_r = self.W_k_r(hidden_states)
        query_states[..., ~self.nope_mask_for_q] = q_r
        key_states[..., ~self.nope_mask] = k_r
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(key_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self.apply_custom_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        query_states = query_states.transpose(1, 2).reshape(bsz, q_len, -1)
        key_states = key_states.transpose(1, 2).reshape(bsz, q_len, -1)
        q_r = query_states[..., ~self.nope_mask_for_q]
        k_r = key_states[..., ~self.nope_mask]

        # prepare c_kv 
        if self.is_share_W_down:
            c_kv = self.W_down_k(hidden_states)
        else:
            c_kv = torch.cat(
                [self.W_down_k(hidden_states), self.W_down_v(hidden_states)], dim=-1
            )
        q_r = q_r.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
        k_r = k_r.view(bsz, q_len, self.num_key_value_heads, -1).transpose(1, 2) # [bsz,q_len,num_key_value_heads,head_rope_dim]
        c_kv = c_kv.view(bsz, q_len, 1, self.low_rank*self.num_key_value_heads).transpose(1, 2) # [bsz,q_len,1 ,self.low_rank * self.num_key_value_heads]

        # prepare q_nope
        q_nope = self.W_down_q(hidden_states)
        q_nope = q_nope.view(
            bsz, q_len, self.num_heads, self.low_rank * self.num_key_value_heads
        ).transpose(1, 2)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_r, c_kv = past_key_value.update(
                k_r,
                c_kv,
                self.layer_idx,
                cache_kwargs,
            )

        query_states = torch.cat([q_nope,q_r],dim=-1)
        # key_states = torch.cat([c_kv,k_r],dim=-1)
        # value_states = c_kv
        return query_states, k_r, c_kv

    def apply_custom_rotary_pos_emb(self, query_states, key_states, cos, sin):
        if self.config.RoPE["partial_rope_version"] == 4:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin, layer_idx=self.layer_idx
            )
        else:
            query_states, key_states = modeling_llama.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        return query_states, key_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if not self.is_inference:
            self.matrix_fusion()

        bsz, q_len, _ = hidden_states.size()
        query_states, k_r, c_kv = self.get_qk_states(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            **kwargs,
        )

        k_r = repeat_kv(k_r, self.num_key_value_groups)
        c_kv = repeat_kv(c_kv, self.num_heads)
        head_rope_dim = (self.nope_mask_for_q == False).sum().item() // self.num_heads
        attn_weights_rope = torch.matmul(query_states[...,-head_rope_dim:], k_r.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights_nope = torch.matmul(query_states[...,:-head_rope_dim], c_kv.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights_rope + attn_weights_nope
        assert attention_mask is not None, "attention_mask is required"
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : k_r.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, c_kv)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value




@staticmethod
def custom_ignore_causal_mask_sdpa(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
    is_training: bool = False,
) -> bool:
    return False


def infer_patch_hf(rope_cfg=None):
    modeling_llama.LLAMA_ATTENTION_CLASSES = {
        "eager": CustomLlamaMLAForInfer,
        "flash_attention_2": CustomLlamaMLAForInfer,
        "sdpa": CustomLlamaMLAForInfer,
    }

    import transformers
    transformers.modeling_attn_mask_utils.AttentionMaskConverter._ignore_causal_mask_sdpa = custom_ignore_causal_mask_sdpa

    if rope_cfg is not None:
        # replace apply_rotary_pos_emb function in llama model
        from ..partial_rope.patch_func_hf import create_custom_apply_rotary_pos_emb_hf

        modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(
            rope_cfg
        )
