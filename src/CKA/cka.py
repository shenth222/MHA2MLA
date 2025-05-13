import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import argparse
from tqdm import tqdm
from IPython import embed
import numpy
import plotly.express as px
import os

def get_parser():
    parser = argparse.ArgumentParser(description="CKA test")
    model_choices = ["135m", "2-7b", "360m", "1b"]
    parser.add_argument("--model_name", type=str, required=True, choices=model_choices, help="Model name for test")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for test")
    parser.add_argument("--figure_path", type=str, default="./figure/CKA/", help="Path to save figure")
    return parser

def load_model(path, config):
    model = AutoModelForCausalLM.from_pretrained(path, config=config)
    return model

def load_config(path):
    config = AutoConfig.from_pretrained(path)
    # config._attn_implementation = 'eager'
    return config

def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

MODEL_MAP = {
    "135m": "/data/shenth/models/SmolLM/135m",
    "360m": "/data/shenth/models/SmolLM/360m",
    "2-7b": "/data/shenth/models/llama/2-7b-hf",
    "1b": "/data/shenth/models/SmolLM/1b"
}

TASK_MAP = {
    "glue": "/data/shenth/datasets/glue",
    "mmlu": "data/shenth/datasets/mmlu"
}

def get_Center_matrix(n):
    return torch.eye(n) - torch.ones((n,n)) / n

def get_Gram_matrix(H, X):
    # X \in \mathbb{R}^{n \times d}, where each of the n rows corresponds to a token
    # X: num_layers, bsz, seq_len, num_kv_heads * hidden_size
    return torch.matmul(torch.matmul(torch.matmul(H, X), X.transpose(-1, -2)), H)

def trace_4_bmm(g1, g2):
    # input matries must be 3D
    # g: bsz, seq_len, seq_len
    return torch.bmm(g1, g2).diagonal(dim1=-2, dim2=-1).sum(dim=-1) 

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    model_name = args.model_name
    task = args.task
    bsz = args.bsz
    figure_path = args.figure_path
    model_path = MODEL_MAP[model_name]
    data_path = TASK_MAP[task]

    config = load_config(model_path)
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, config).cuda().eval()
    
    prompt = ["Hey, are you conscious? Can you talk to me?",
              "How are you?",
              "Today is good day!",
              "What are you doing Mike?"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    res = model.generate(inputs.input_ids.cuda(), max_length=20, return_dict_in_generate=True, use_cache=True)
    # res['past_key_values'] # num_layers, K/V, tensor[bsz, num_kv_heads, seq_len, hidden_size]
    key_states = torch.stack([k[0] for k in res['past_key_values']]) # num_layers, bsz, num_kv_heads, seq_len, hidden_size
    value_states = torch.stack([k[1] for k in res['past_key_values']])
    
    num_layers, bsz, _, seq_len, _ = key_states.shape
    H = get_Center_matrix(seq_len).cuda()
    G = get_Gram_matrix(H, key_states.transpose(2,3).reshape(num_layers, bsz, seq_len, -1)) # num_layers, bsz, seq_len, seq_len

    # traces = G.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # shape: (num_layers, bsz)
    # torch.trace(torch.matmul(g,g))

    # trace(G^2)
    g2 = (G @ G).diagonal(dim1=-2, dim2=-1).sum(dim=-1) # num_layers, bsz

    # trace(G1*G2)
    gg = [] # a lower triangular matrix # num_layers_row, num_layers_col, bsz
    for layer_idx in range(num_layers):
        if layer_idx+1 == num_layers: break
        gg.append(torch.stack([trace_4_bmm(G[layer_idx], G[idx]) for idx in range(layer_idx+1, num_layers)]))

    # cka
    cka = torch.eye(num_layers)
    for layer_idx in range(num_layers):
        if layer_idx+1 == num_layers: break
        for idx in range(layer_idx+1, num_layers):
            cka[layer_idx][idx] = (gg[layer_idx][idx-layer_idx-1] / torch.sqrt(g2[layer_idx]*g2[idx])).mean()
    # padding to a full matrix
    cka = cka + cka.T - torch.eye(num_layers)

    # heatmap
    data = cka.cpu().numpy()
    fig = px.imshow(
        data,
        labels=dict(x="layers", y="layers", color="Value"),
        x=list(range(num_layers)),
        y=list(range(num_layers)),
        color_continuous_scale=["#80a6c3", "#ffffff", "#da3b46"]
    )
    os.makedirs("./figure", exist_ok=True)
    fig.write_image("./figure/test.png")
    # embed()





