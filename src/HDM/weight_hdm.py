import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from IPython import embed
from loguru import logger
import argparse
import plotly.express as px
from utils import load_HDM_res, load_model
import sys
sys.path.append("/data/shenth/work/HadamardDecompositions/")
from Hadamard_BCD import Hadamard_BCD, Hadamard_BCD_torch
from tqdm import tqdm, trange
import os
sys.path.append("/data/shenth/work/MHA2MLA/src/")
from CKA.cka_utils import linear_cka_centered_torch

MODE = ["Q", "K", "V", "O", "U", "D", "G"]
# RANK = 256

def HDM_for_weight(model, mode="Q", rank=32):
    assert mode in MODE, f"current mode: {mode} is not in MODE: {MODE}"
    logger.info(f"Hadamard for {mode} weights")
    W1, H1, W2, H2 = [], [], [], []
    num_layers = len(model.model.layers)

    def get_weights(layer_idx):
        if mode == "Q":
            return model.model.layers[layer_idx].self_attn.q_proj.weight.detach()
        elif mode == "K":
            return model.model.layers[layer_idx].self_attn.k_proj.weight.detach()
        elif mode == "V":
            return model.model.layers[layer_idx].self_attn.v_proj.weight.detach()
        elif mode == "O":
            return model.model.layers[layer_idx].self_attn.o_proj.weight.detach()
        elif mode == "U":
            return model.model.layers[layer_idx].mlp.up_proj.weight.detach().T
        elif mode == "D":
            return model.model.layers[layer_idx].mlp.down_proj.weight.detach()
        else:
            return model.model.layers[layer_idx].mlp.gate_proj.weight.detach().T

    
    tbar = tqdm(range(num_layers))
    for layer_idx in range(num_layers):
        tbar.update(1)
        tbar.set_description(f"Computing layer {layer_idx+1}")
        tbar.refresh()
        M = get_weights(layer_idx).cuda()
        a, b, c, d, _, _ = Hadamard_BCD_torch(M, rank)
        W1.append(a)
        H1.append(b)
        W2.append(c)
        H2.append(d)
    
    return W1, H1, W2, H2

def save_HDM_res(W1, H1, W2, H2, save_path="./HDM_res", mode="Q", rank=512):
    assert W1 is not None, "W1 is None"
    assert H1 is not None, "H1 is None"
    assert W2 is not None, "W2 is None"
    assert H2 is not None, "H2 is None"
    save_path = f"{save_path}/{mode}"
    os.makedirs(save_path, exist_ok=True)
    path = f"{save_path}/W1_{mode}_{rank}.pkl"
    torch.save(W1, path)
    path = f"{save_path}/H1_{mode}_{rank}.pkl"
    torch.save(H1, path)
    path = f"{save_path}/W2_{mode}_{rank}.pkl"
    torch.save(W2, path)
    path = f"{save_path}/H2_{mode}_{rank}.pkl"
    torch.save(H2, path)

def cka_by_layer(input):
    assert input is not None, "Input is None"
    num_layers = len(input)
    cka_matrix = torch.zeros(num_layers, num_layers)
    for i in trange(num_layers):
        for j in range(num_layers):
            cka_matrix[i, j] = linear_cka_centered_torch(input[i], input[j])
    return cka_matrix

def cossim_by_layer(input):
    assert input is not None, "Input is None"
    num_layers = len(input)
    sim_matrix = torch.zeros(num_layers, num_layers)
    for i in trange(num_layers):
        for j in range(num_layers):
            matrix1 = input[i].reshape(-1)
            matrix2 = input[j].reshape(-1)
            cos_sim = torch.dot(matrix1, matrix2) / (torch.norm(matrix1) * torch.norm(matrix2))
            sim_matrix[i, j] = cos_sim.item()
    return sim_matrix

def get_parser():
    parser = argparse.ArgumentParser(description="Benchmark for cka")
    parser.add_argument("--mode", type=str, required=True, choices=MODE, help="Q/K/V/O which?")
    parser.add_argument("--rank", type=int, required=True)
    return parser

def make_heatmap(tensor, title, color_continuous_scale=None, x_label="Layer", y_label="Layer", fig_path="./figure/cka/", file_name="", zmin=0, zmax=1):
    matrix = tensor.detach().cpu().numpy()
    num_layers = tensor.shape[0]
    fig = px.imshow(
        matrix,
        labels=dict(x=x_label, y=y_label, color="similarity"),
        x=list(range(num_layers)),
        y=list(range(num_layers)),
        color_continuous_scale=color_continuous_scale,
        zmin=zmin,
        zmax=zmax
    )
    fig.update_layout(title=title, title_font_size=20)
    os.makedirs(fig_path, exist_ok=True)
    fig.write_image(f"{fig_path}{file_name}.png")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = AutoConfig.from_pretrained("/data/shenth/models/llama/2-7b-hf")
    config._attn_implementation = "eager"
    model = load_model(config)
    mode = args.mode
    RANK = args.rank
    W1, H1, W2, H2 = HDM_for_weight(model, mode, RANK)
    save_HDM_res(W1, H1, W2, H2, mode=mode, rank=RANK)

    # W1, H1, W2, H2 = load_HDM_res(mode=mode, rank=RANK)

    # CKA
    W1_cka = cka_by_layer(W1)
    H1_cka = cka_by_layer(H1)
    W2_cka = cka_by_layer(W2)
    H2_cka = cka_by_layer(H2)
    color_continuous_scale = [[0,"#80a6c3"], [0.5,"#ffffff"], [1,"#da3b46"]]
    make_heatmap(W1_cka, title=f"CKA W1_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, file_name=f"cka_W1_{mode}_{RANK}")
    make_heatmap(H1_cka, title=f"CKA H1_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, file_name=f"cka_H1_{mode}_{RANK}")
    make_heatmap(W2_cka, title=f"CKA W2_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, file_name=f"cka_W2_{mode}_{RANK}")
    make_heatmap(H2_cka, title=f"CKA H2_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, file_name=f"cka_H2_{mode}_{RANK}")

    # Cossim
    W1_sim = cossim_by_layer(W1)
    H1_sim = cossim_by_layer(H1)
    W2_sim = cossim_by_layer(W2)
    H2_sim = cossim_by_layer(H2)
    color_continuous_scale = [[0,"#80a6c3"], [0.5,"#ffffff"], [1,"#da3b46"]]
    make_heatmap(W1_sim, title=f"Sim W1_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, fig_path="./figure/sim/", file_name=f"sim_W1_{mode}_{RANK}", zmin=-1, zmax=1)
    make_heatmap(H1_sim, title=f"Sim H1_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, fig_path="./figure/sim/", file_name=f"sim_H1_{mode}_{RANK}", zmin=-1, zmax=1)
    make_heatmap(W2_sim, title=f"Sim W2_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, fig_path="./figure/sim/", file_name=f"sim_W2_{mode}_{RANK}", zmin=-1, zmax=1)
    make_heatmap(H2_sim, title=f"Sim H2_{mode}_{RANK}", color_continuous_scale=color_continuous_scale, fig_path="./figure/sim/", file_name=f"sim_H2_{mode}_{RANK}", zmin=-1, zmax=1)
    




