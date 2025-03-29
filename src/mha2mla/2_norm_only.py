from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from patch_llama_attn import patch_llama_attn, get_qk_embed_states, get_qk_states
import math

# path = "/data/shenth/models/SmolLM/135m"
path = "/data/shenth/models/llama/2-7b-hf"

patch_llama_attn()

config = AutoConfig.from_pretrained(path)
config._attn_implementation = 'eager'
tokenizer = AutoTokenizer.from_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path, config=config)
model.eval().to("cuda")

head_dim = config.hidden_size // config.num_attention_heads
num_layers = config.num_hidden_layers
group_size = config.num_attention_heads // config.num_key_value_heads
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs)

del model

q, k = get_qk_states() # [layer_idx, bsz, num_heads, seq_len, head_dim]
q_embed, k_embed = get_qk_embed_states() # [layer_idx, bsz, num_key_value_heads, seq_len, head_dim]
# layer_idx cycles in groups of {num_layers}

assert q, "sth wrong with q & k"

# ensure seq_len = 1
# q = q[num_layers:]
# k = k[num_layers:]
# q_embed = q_embed[num_layers:]
# k_embed = k_embed[num_layers:]

# # [layer_idx, bsz, num_heads(num_key_value_heads), head_dim]
# q = [x.squeeze(3) for x in q]
# k = [x.squeeze(3) for x in k]
# q_embed = [x.squeeze(3) for x in q_embed]
# k_embed = [x.squeeze(3) for x in k_embed]

def cal_2_norm(states):
    # [bsz, num_heads(num_key_value_heads), seq_len, head_dim]
    states = states.reshape(states.shape[0], states.shape[1], states.shape[2], 2, -1).transpose(-1, -2)
    # [bsz, num_heads(num_key_value_heads), seq_len, head_dim//2, 2]
    states = torch.norm(states, p=2, dim=4)
    # [bsz, num_heads(num_key_value_heads), seq_len, head_dim//2]
    states = states.mean(dim=2, keepdim=False)
    # [bsz, num_heads(num_key_value_heads), head_dim//2]
    states = states.mean(dim=0, keepdim=False)
    # [num_heads(num_key_value_heads), head_dim//2]
    return states

q = [cal_2_norm(x) for x in q] # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
k = [cal_2_norm(x) for x in k]
q_embed = [cal_2_norm(x) for x in q_embed]
k_embed = [cal_2_norm(x) for x in k_embed]

q = torch.stack(q, dim=0) # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
k = torch.stack(k, dim=0)
q_embed = torch.stack(q_embed, dim=0)
k_embed = torch.stack(k_embed, dim=0)

q = q.reshape(num_layers, -1, num_heads, head_dim//2) # [layer_idx, cycles, num_heads(num_key_value_heads), head_dim//2]
k = k.reshape(num_layers, -1, num_kv_heads, head_dim//2)
q_embed = q_embed.reshape(num_layers, -1, num_heads, head_dim//2)
k_embed = k_embed.reshape(num_layers, -1, num_kv_heads, head_dim//2)

q = torch.mean(q, dim=1, keepdim=False) # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
k = torch.mean(k, dim=1, keepdim=False)
q_embed = torch.mean(q_embed, dim=1, keepdim=False)
k_embed = torch.mean(k_embed, dim=1, keepdim=False)

import plotly.express as px
def make_heatmap(states, flag, num_heads, num_layers):
    for layer_idx in range(num_layers):
        data = states[layer_idx].transpose(0,1).cpu().numpy() # head_dim//2, num_heads(num_key_value_heads)

        fig = px.imshow(
            data,
            labels=dict(x="num_heads", y="frequency", color="Value"),
            x=list(range(num_heads)),
            title=f"Layer {layer_idx} - {flag} states",
        )

        # 隐藏 y 轴刻度但保留 y 轴标签
        fig.update_layout(yaxis=dict(showticklabels=False))

        fig.write_image(f"./figure/{flag}/layer_{layer_idx}_{flag}_states.png", scale=3)

def show_heatmap():
    make_heatmap(q, "q", num_heads, num_layers)
    make_heatmap(k, "k", num_kv_heads, num_layers)
    make_heatmap(q_embed, "q_embed", num_heads, num_layers)
    make_heatmap(k_embed, "k_embed", num_kv_heads, num_layers)

# from IPython import embed
# embed()

# diff nope and rope
def show_diff():
    diff_q = torch.abs(q - q_embed)
    diff_k = torch.abs(k - k_embed)
    make_heatmap(diff_q, "diff_q", num_heads, num_layers)
    make_heatmap(diff_k, "diff_k", num_kv_heads, num_layers)

show_heatmap()
show_diff()


    

