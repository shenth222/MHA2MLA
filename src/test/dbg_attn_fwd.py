import torch
from transformers import LlamaConfig as LlamaConfig_hf
from nanotron.config import LlamaConfig as LlamaConfig_nt
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv


import argparse
import os
import math
from pathlib import Path
from torch import nn

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
    decode_tokenized,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

from ..mla.mla_patch_hf import CustomLlamaSdpaAttention,mla_patch_hf,CustomLlamaAttention,CustomLlamaFlashAttention2
from ..mla.mla_patch_nt import CustomCausalSelfAttention,mla_patch_nt
torch.manual_seed(42)  # 114514 是随机种子值，可以替换为任意整数



def undo_repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    Undo the effect of repeat_kv. This function reshapes the hidden states from
    (batch, num_attention_heads, seqlen, head_dim) back to (batch, num_key_value_heads, seqlen, head_dim).
    """
    batch, num_attention_heads, slen, head_dim = hidden_states.shape
    # 计算原始的 num_key_value_heads
    num_key_value_heads = num_attention_heads // num_key_value_groups
    # 将 hidden_states 重塑为 (batch, num_key_value_heads, num_key_value_groups, slen, head_dim)
    hidden_states = hidden_states.reshape(batch, num_key_value_heads, num_key_value_groups, slen, head_dim)
    # 取每个组的第一个元素（因为重复的值是相同的）
    hidden_states = hidden_states[:, :, 0, :, :]
    return hidden_states


# 实例化
config_hf=LlamaConfig_hf(
    vocab_size=32000,
    hidden_size=576,
    intermediate_size=12*4,
    num_hidden_layers=1,
    num_attention_heads=9,
    num_key_value_heads=3,
    hidden_act="silu",
    max_position_embeddings=2048,
    initializer_range=0.02,
    rms_norm_eps=1e-6,
    use_cache=False,
    pad_token_id=None,
    bos_token_id=1,
    eos_token_id=2,
    pretraining_tp=1,
    tie_word_embeddings=False,
    rope_theta=10000.0,
    rope_scaling=None,
    attention_bias=False,
    attention_dropout=0.0,
    mlp_bias=False,
    head_dim=None,
    RoPE={
        "partial_rope_version": 4,  # 0 1 2 3 4 5
        "top_k_rope_dim": 4,
        "last_k_rope_dim": 0,
        "uniform_start_point": 0,
        "uniform_step": 4,
        "qk_tensor_path": '/home/binguo/data/MLA-FT/utils/qk_tensor_135M.pth',
        "n_gqa_group": 3,
    },
    SVD={
        "method": 2,
        "low_rank": 8,
    },
    attn_implementation="flash_attention_2",
)
llama_config_params = {
    "bos_token_id": config_hf.bos_token_id,
    "eos_token_id": config_hf.eos_token_id,
    "hidden_act": config_hf.hidden_act,
    "hidden_size": config_hf.hidden_size,
    "initializer_range": config_hf.initializer_range,
    "intermediate_size": config_hf.intermediate_size,
    "max_position_embeddings": config_hf.max_position_embeddings,
    "num_attention_heads": config_hf.num_attention_heads,
    "num_hidden_layers": config_hf.num_hidden_layers,
    "num_key_value_heads": config_hf.num_key_value_heads,
    "pad_token_id": config_hf.pad_token_id,
    "pretraining_tp": config_hf.pretraining_tp,
    "rms_norm_eps": config_hf.rms_norm_eps,
    "rope_scaling": config_hf.rope_scaling,
    "rope_theta": config_hf.rope_theta,
    "rope_interleaved": False,  # 默认值
    "tie_word_embeddings": config_hf.tie_word_embeddings,
    "use_cache": config_hf.use_cache,
    "vocab_size": config_hf.vocab_size,
}
torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)
mla_patch_hf(config_hf.RoPE)
mla_patch_nt(config_hf.RoPE)


# 创建 LlamaConfig 实例
config_llama = LlamaConfig_nt(**llama_config_params)
setattr(config_llama, "RoPE", config_hf.RoPE)
setattr(config_llama, "SVD", config_hf.SVD)
attn_hf=CustomLlamaFlashAttention2(config_hf,layer_idx=0)
parallel_config = ParallelismArgs(
    dp=1,
    pp=1,
    tp=1,
    pp_engine=OneForwardOneBackwardPipelineEngine(),
    tp_mode=TensorParallelLinearMode.ALL_REDUCE,
    tp_linear_async_communication=False,
)
parallel_context = ParallelContext(
    data_parallel_size=parallel_config.dp,
    pipeline_parallel_size=parallel_config.pp,
    tensor_parallel_size=parallel_config.tp,
)
attn_nt=CustomCausalSelfAttention(config_llama,layer_idx=0,parallel_config=parallel_config,tp_pg=parallel_context.tp_pg)

attn_hf.load_state_dict(torch.load("state_dict.pth"))
attn_nt.load_state_dict(torch.load("state_dict.pth"))

# 输入数据
import pickle
fwd_param_dict=pickle.load(open("forward_args_hf.pkl","rb"))
# 转换为下三角矩阵
inputs_nt = fwd_param_dict["hidden_states"].transpose(0, 1)
mask_nt = torch.ones((inputs_nt.size(1), inputs_nt.size(0)), dtype=torch.bool)


fwd_param_dict["attnetion_mask"]=mask_nt
ouputs_hf=attn_hf(
    hidden_states=fwd_param_dict["hidden_states"],
    attention_mask=fwd_param_dict["attention_mask"],
    position_ids=fwd_param_dict["position_ids"],
    past_key_value=fwd_param_dict["past_key_value"],
    output_attentions=fwd_param_dict["output_attentions"],
    use_cache=fwd_param_dict["use_cache"],
    position_embeddings=fwd_param_dict["position_embeddings"],
)

# ouputs_hf=attn_hf(**fwd_param_dict)
ouputs_nt=attn_nt(inputs_nt,mask_nt)
gqa_size=config_hf.num_attention_heads//config_hf.num_key_value_heads
bsz, q_len, _ = fwd_param_dict["hidden_states"].shape

# q_hf,k_hf,v_hf=ouputs_hf[-1] # [bsz,q_len,num_heads,head_dim]
# k_hf=undo_repeat_kv(k_hf,gqa_size)
# v_hf=undo_repeat_kv(v_hf,gqa_size)
# q_nt,k_nt,v_nt=ouputs_nt["qkv"]
# q_nt=q_nt.reshape(bsz,q_len,config_hf.num_attention_heads,-1).transpose(1,2)
# k_nt=k_nt.reshape(bsz,q_len,config_hf.num_key_value_heads,-1).transpose(1,2)
# v_nt=v_nt.reshape(bsz,q_len,config_hf.num_key_value_heads,-1).transpose(1,2)
# print("q")
# print(torch.allclose(q_hf,q_nt))
# print("k")
# print(torch.allclose(k_hf,k_nt))
# print("v")
# print(torch.allclose(v_hf,v_nt))
# print("attn_output:")
attn_output_nt=ouputs_nt["hidden_states"].transpose(0,1)
attn_output_hf=ouputs_hf[0]
torch.save(attn_output_hf,"attn_output_hf.pth")
torch.save(attn_output_nt,"attn_output_nt.pth")
print(torch.allclose(attn_output_nt,attn_output_hf))
print(torch.allclose(torch.load("attn_output_hf.pth"),torch.load("attn_output_nt.pth")))
print()
abs_diff = torch.abs(attn_output_nt - attn_output_hf)
print("绝对差值的最大值:", torch.max(abs_diff).item())
abs_ref = torch.maximum(torch.abs(attn_output_hf), torch.abs(attn_output_nt))
relative_diff = abs_diff / torch.maximum(abs_ref, torch.tensor(1e-5))
max_relative_diff = torch.max(relative_diff)
print("相对比例的最大值:", max_relative_diff.item())
print(attn_output_hf[3, 1769, 397])
print(attn_output_nt[3, 1769, 397])