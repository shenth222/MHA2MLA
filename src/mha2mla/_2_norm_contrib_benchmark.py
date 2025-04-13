from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from patch_llama_attn import patch_llama_attn, get_qk_embed_states, get_qk_states, clear_qk_states
import math
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
import plotly.express as px
import random
patch_llama_attn()

LOAD_STATES = False
SAVE_STATES = True
if LOAD_STATES:
    SAVE_STATES = False

model_name = "2-7b" # "135m", "2-7b", "360m"
path = "/data/shenth/models/llama/2-7b-hf"

def load_model(path, config):
    model = LlamaForCausalLM.from_pretrained(path, config=config)
    return model

def load_config(path):
    config = AutoConfig.from_pretrained(path)
    config._attn_implementation = 'eager'
    return config

def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def make_datasets(data_path, tokenizer, bsz=8):
    dataset = load_dataset(data_path, "ax")
    ml = max(len(tokenizer(x["premise"])["input_ids"]) for x in dataset["test"])

    # 对数据进行填充和截断
    def preprocess_with_padding(examples):
        return tokenizer(
            examples["premise"],
            max_length=ml,  # 使用最大长度
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    encoded_dataset = dataset["test"].map(preprocess_with_padding, batched=True)
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # 创建 DataLoader
    test_dataset = encoded_dataset
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=bsz,
        )
    return test_dataloader

def save_qk_states(states, flag, model_name):
    # need fix bugs
    # [layer_idx, num_heads(num_key_value_heads), head_dim]
    if model_name not in ["135m", "2-7b", "360m"]:
        raise ValueError("model_name should be in [135m, 2-7b, 360m]")
    save_dir = f"./qk_state/states/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(states, f"{save_dir}/{flag}.pt")

if LOAD_STATES:
    # need fix bugs
    pass
else:
    config = load_config(path)
    tokenizer = load_tokenizer(path)
    model = load_model(path, config)
    # datasets
    data_path = "/data/shenth/datasets/glue"
    test_dataloader = make_datasets(data_path, tokenizer, bsz=1)
    model.eval().to("cuda")
    q, k = [], []
    