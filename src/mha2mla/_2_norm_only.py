from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from patch_llama_attn import patch_llama_attn, get_qk_embed_states, get_qk_states, clear_qk_states
import math
from tqdm import tqdm
import os

model_name = "2-7b" # "135m", "2-7b", "360m"
# path = "/data/shenth/models/SmolLM/135m"
path = "/data/shenth/models/llama/2-7b-hf"

LOAD_STATES = True

patch_llama_attn()

config = AutoConfig.from_pretrained(path)
config._attn_implementation = 'eager'
tokenizer = AutoTokenizer.from_pretrained(path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

head_dim = config.hidden_size // config.num_attention_heads
num_layers = config.num_hidden_layers
group_size = config.num_attention_heads // config.num_key_value_heads
num_heads = config.num_attention_heads
num_kv_heads = config.num_key_value_heads

# datasets
from datasets import load_dataset
data_path = "/data/shenth/datasets/glue"
dataset = load_dataset(data_path, "ax")
# 获取 "test" 集中 "premise" 列的最大长度
lengths = [len(premise) for premise in dataset["test"]["premise"]]
ml = max(lengths)

def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        max_length=ml,
        padding="max_length",
        truncation=True,
        return_tensors="pt")

encoded_dataset = dataset["test"].map(preprocess_function, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

from torch.utils.data import DataLoader

# 加载
test_dataset = encoded_dataset

# 创建 DataLoader
from torch.utils.data import DataLoader
test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    )

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

def get_one_iteration_2_norm():
    q, k = get_qk_states() # [layer_idx, bsz, num_heads, seq_len, head_dim]
    q_embed, k_embed = get_qk_embed_states() # [layer_idx, bsz, num_key_value_heads, seq_len, head_dim]
    # layer_idx cycles in groups of {num_layers}
    assert q, "sth wrong with q & k"

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

    return q, k, q_embed, k_embed

import plotly.express as px
def make_heatmap(states, flag, num_heads, num_layers, model_name):
    if model_name not in ["135m", "2-7b", "360m"]:
        raise ValueError("model_name should be in [135m, 2-7b, 360m]")
    for layer_idx in range(num_layers):
        data = states[layer_idx].transpose(0,1).cpu().numpy() # head_dim//2, num_heads(num_key_value_heads)

        fig = px.imshow(
            data,
            labels=dict(x="num_heads", y="frequency", color="Value"),
            x=list(range(num_heads)),
            y=list(range(data.shape[0])),
            title=f"Layer {layer_idx} - {flag} states",
            color_continuous_scale="YlGnBu",
            aspect="auto",
        )

        # 隐藏 y 轴刻度但保留 y 轴标签
        fig.update_layout(
            xaxis=dict(
                showticklabels=True,  # 显示 x 轴刻度
            ),
            yaxis=dict(
                showticklabels=False,  # 隐藏 y 轴刻度
            ),
            width=600,
            height=600,
        )
        
        save_dir = f"./figure/{model_name}/{flag}/"
        os.makedirs(save_dir, exist_ok=True)

        fig.write_image(f"{save_dir}/layer_{layer_idx}_{flag}_states.png", scale=3)

def make_heatmap_2(states, flag, num_heads, num_layers, model_name):
    if model_name not in ["135m", "2-7b", "360m"]:
        raise ValueError("model_name should be in [135m, 2-7b, 360m]")
    data = states.transpose(0,1).cpu().numpy() # head_dim//2, layer_idx
    fig = px.imshow(
        data,
        labels=dict(x="layer_idx", y="frequency", color="Value"),
        x=list(range(num_layers)),
        y=list(range(data.shape[0])),
        title=f"{flag} states",
        color_continuous_scale="YlGnBu",
        aspect="auto",
    )
    fig.update_layout(
        xaxis=dict(
            showticklabels=True,  # 显示 x 轴刻度
        ),
        yaxis=dict(
            showticklabels=False,  # 隐藏 y 轴刻度
        ),
        width=600,
        height=600,
    )
    save_dir = f"./figure/{model_name}/{flag}/"
    os.makedirs(save_dir, exist_ok=True)
    fig.write_image(f"{save_dir}/{flag}_states.png", scale=3)

def save_qk_states(states, flag, model_name):
    # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
    if model_name not in ["135m", "2-7b", "360m"]:
        raise ValueError("model_name should be in [135m, 2-7b, 360m]")
    save_dir = f"./qk_state/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(states, f"{save_dir}/{flag}.pt")

def load_qk_states(model_name):
    load_dir = f"./qk_state/{model_name}/"
    q_ne = torch.load(f"{load_dir}/q_ne.pt")
    k_ne = torch.load(f"{load_dir}/k_ne.pt")
    q_pe = torch.load(f"{load_dir}/q_pe.pt")
    k_pe = torch.load(f"{load_dir}/k_pe.pt")
    return q_ne, k_ne, q_pe, k_pe

# diff nope and rope
def show_diff(q, k, q_embed, k_embed, model_name):
    # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
    diff_q = torch.abs(q - q_embed)
    diff_k = torch.abs(k - k_embed)
    make_heatmap(diff_q, "diff_q", num_heads, num_layers, model_name)
    make_heatmap(diff_k, "diff_k", num_kv_heads, num_layers, model_name)

def show_qk_w_layer(q_ne, k_ne, q_pe, k_pe, model_name):
    # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
    q_ne = q_ne.mean(dim=1, keepdim=False) # [layer_idx, head_dim//2]
    k_ne = k_ne.mean(dim=1, keepdim=False)
    q_pe = q_pe.mean(dim=1, keepdim=False)
    k_pe = k_pe.mean(dim=1, keepdim=False)
    make_heatmap_2(q_ne, "q_ne", num_heads, num_layers, model_name)
    make_heatmap_2(k_ne, "k_ne", num_kv_heads, num_layers, model_name)
    make_heatmap_2(q_pe, "q_pe", num_heads, num_layers, model_name)
    make_heatmap_2(k_pe, "k_pe", num_kv_heads, num_layers, model_name)

# test
if LOAD_STATES:
    q_ne, k_ne, q_pe, k_pe = load_qk_states(model_name)
    if type(q_ne) == list:
        q_ne = torch.stack(q_ne, dim=0)
        k_ne = torch.stack(k_ne, dim=0)
        q_pe = torch.stack(q_pe, dim=0)
        k_pe = torch.stack(k_pe, dim=0)
    show_qk_w_layer(q_ne, k_ne, q_pe, k_pe, model_name)

else:
    model = LlamaForCausalLM.from_pretrained(path, config=config)
    model.eval().to("cuda")
    q_ne, k_ne, q_pe, k_pe = [], [], [], []
    for batch in tqdm(test_dataloader, desc="Processing batches", total=len(test_dataloader)):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        clear_qk_states()
        q, k = get_qk_states()
        if q is not None:
            AssertionError("sth wrong with q & k")
        with torch.no_grad():
            model(**batch)
            q, k, q_embed, k_embed = get_one_iteration_2_norm() # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
            q_ne.append(q)
            k_ne.append(k)
            q_pe.append(q_embed)
            k_pe.append(k_embed)

    q_ne = torch.stack(q_ne, dim=0) # [bsz, layer_idx, num_heads(num_key_value_heads), head_dim//2]
    k_ne = torch.stack(k_ne, dim=0)
    q_pe = torch.stack(q_pe, dim=0)
    k_pe = torch.stack(k_pe, dim=0)
    q_ne = q_ne.mean(dim=0, keepdim=False) # [layer_idx, num_heads(num_key_value_heads), head_dim//2]
    k_ne = k_ne.mean(dim=0, keepdim=False)
    q_pe = q_pe.mean(dim=0, keepdim=False)
    k_pe = k_pe.mean(dim=0, keepdim=False)

    save_qk_states(q_ne, "q_ne", model_name)
    save_qk_states(k_ne, "k_ne", model_name)
    save_qk_states(q_pe, "q_pe", model_name)
    save_qk_states(k_pe, "k_pe", model_name)

# q_ne, k_ne, q_pe, k_pe = load_qk_states(model_name)

# make_heatmap(q_ne, "q_ne", num_heads, num_layers, model_name)
# make_heatmap(k_ne, "k_ne", num_kv_heads, num_layers, model_name)
# make_heatmap(q_pe, "q_pe", num_heads, num_layers, model_name)
# make_heatmap(k_pe, "k_pe", num_kv_heads, num_layers, model_name)
# show_diff(q_ne, k_ne, q_pe, k_pe, model_name)

