from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig, DataCollatorForLanguageModeling
import torch
import argparse
from transformers.models.llama.modeling_llama import LlamaAttention
from patch_llama_attn import patch_llama_attn, get_qk_embed_states, get_qk_states, clear_qk_states
import math
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
import plotly.express as px
import random
from IPython import embed
patch_llama_attn()

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


def save_qk_states(states, flag, model_name):
    # need fix bugs
    # [layer_idx, num_heads(num_key_value_heads), head_dim]
    if model_name not in ["135m", "2-7b", "360m"]:
        raise ValueError("model_name should be in [135m, 2-7b, 360m]")
    save_dir = f"./qk_state/states/{model_name}/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(states, f"{save_dir}/{flag}.pt")


MODEL_MAP = {
    "135m": "/data/shenth/models/SmolLM/135m",
    "360m": "/data/shenth/models/SmolLM/135m",
    "2-7b": "/data/shenth/models/llama/2-7b-hf"
}

DATASET_MAP = {
    "mmlu": "/data/shenth/datasets/mmlu",
    "glue": "/data/shenth/datasets/glue"
}

if __name__ == "__main__":
    model_path = MODEL_MAP["135m"]
    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, config).cuda()
    text = ["Hello, my name is jack. What is your"]
    inputs = tokenizer(text, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids.to("cuda"), max_new_tokens=2)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(output)