import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from IPython import embed
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import plotly.express as px
import os

model_name = "/data/shenth/models/llama/2-7b-hf"  # Or any other Llama checkpoint available

def load_and_process_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    return model

model = load_and_process_model()
num_layers = len(model.model.layers)

def linear_cka_centered_torch(kv1: torch.Tensor, kv2: torch.Tensor) -> torch.Tensor:
    """
    A *centered* linear CKA, as in Kornblith et al. (2019), for (L, D) Tensors.
    This subtracts each row's mean from kv1, kv2 before computing the norm-based formula.
    
    Steps:
      1. Row-center each representation (i.e., subtract column means).
      2. Compute Frobenius norms of X^T X, Y^T Y, X^T Y on the centered data.
      3. Return (||X^T Y||_F^2) / (||X^T X||_F * ||Y^T Y||_F).

    Note:
      - 'Row-center' means we subtract the *column* mean for each dimension (the usual approach 
        in CKA references). This ensures the average vector over all tokens is zero.

    Args:
      kv1: shape (L, D)
      kv2: shape (L, D)

    Returns:
      cka_value: a scalar torch.Tensor
    """
    assert kv1.shape[1] == kv2.shape[1], "kv1, kv2 must have same embedding dimension."

    # Move to GPU if desired
    device = kv1.device
    kv1 = kv1.to(device)
    kv2 = kv2.to(device)
    
    # 1. Row-center each representation. 
    #    (Compute column means & subtract => each dimension has mean 0 across L)
    kv1_centered = kv1 - kv1.mean(dim=0, keepdim=True)
    kv2_centered = kv2 - kv2.mean(dim=0, keepdim=True)
    
    # 2. Norm computations
    xtx = (kv1_centered.T @ kv1_centered).norm(p='fro')
    yty = (kv2_centered.T @ kv2_centered).norm(p='fro')
    xty = (kv1_centered.T @ kv2_centered).norm(p='fro')

    # Handle degenerate case
    if xtx == 0 or yty == 0:
        return torch.tensor(0.0, device=device, dtype=kv1.dtype)

    # 3. Linear CKA formula
    cka_value = (xty ** 2) / (xtx * yty)

    return cka_value

def plot_heatmap(tensor, title="Heatmap", custom_colors=None, colorbar=True, 
                 x_label="", y_label="", font_size=14, tick_font_size=12, interpolation="bilinear"):
    """
    Plots a smooth heatmap from a 2D PyTorch tensor with a custom color gradient.

    Args:
        tensor (torch.Tensor): 2D tensor to plot.
        title (str): Title of the heatmap.
        custom_colors (list): List of 6 colors for the colormap.
        colorbar (bool): Whether to show the colorbar.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        font_size (int): Font size for title and labels.
        tick_font_size (int): Font size for x and y ticks.
        interpolation (str): Interpolation method for smooth transitions (e.g., 'bilinear', 'bicubic').
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    if tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D.")
    
    # Convert tensor to numpy
    matrix = tensor.detach().cpu().numpy()

    # Define a custom colormap
    # if custom_colors and len(custom_colors) >= 2:
    #     cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_colors, N=256)
    # else:
    #     cmap = "coolwarm"  # Default colormap if none provided

    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    #cmap = sns.color_palette("vlag", as_cmap=True)
    #cmap = 'coolwarm'
    # Plot heatmap with smooth transitions
    plt.figure(figsize=(7, 6))
    im = plt.imshow(matrix, cmap=cmap, aspect='auto')

    # Set title and labels
    plt.title(title, fontsize=23)
    plt.xlabel(x_label, fontsize=21)
    plt.ylabel(y_label, fontsize=21)
    
    # Configure tick parameters
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)

    # Configure colorbar
    if colorbar:
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=19)
    plt.savefig(f"{title}.png", bbox_inches='tight')

cka_matrix = torch.zeros(num_layers, num_layers)
for i in range(num_layers):
    for j in range(num_layers):
        a = model.model.layers[i].self_attn.k_proj.weight.detach()
        b = model.model.layers[j].self_attn.k_proj.weight.detach()
        cka_matrix[i, j] = linear_cka_centered_torch(a, b)
        print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")

cmap = sns.diverging_palette(240, 10)
custom_colors = ["#D93F49", "#E28187", "#EBBFC2", "#D5E1E3", "#AFC9CF", "#8FB4BE"]  
custom_colors = custom_colors[::-1]  # Reverse the colors for better contrast
plot_heatmap(cka_matrix, title=f"CKA-weight Matrix", colorbar=True, x_label="Layer", y_label="Layer", custom_colors=custom_colors)
color_continuous_scale = [[0,"#80a6c3"], [0.5,"#ffffff"], [1,"#da3b46"]]