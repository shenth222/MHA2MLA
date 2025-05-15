import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from IPython import embed

model_name = "/data/shenth/models/llama/2-7b-hf"  # Or any other Llama checkpoint available
tokenizer = AutoTokenizer.from_pretrained(model_name)
def load_and_process_model():
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    model = model.cuda()

    # We'll collect outputs for all layers in these lists
    collected_k_outputs = []
    collected_v_outputs = []
    def k_proj_hook(module, input, output):
        """
        module: The layer that produced this output (k_proj).
        input:  The input to k_proj.
        output: The output from k_proj (shape [batch_size, seq_len, hidden_dim]).
        """
        # Detach to avoid growing the autograd graph
        collected_k_outputs.append(output.detach().cpu())

    def v_proj_hook(module, input, output):
        """
        Same logic as k_proj_hook, but for v_proj.
        """
        collected_v_outputs.append(output.detach().cpu())

    num_layers = len(model.model.layers)
    hooks_k = []
    hooks_v = []
    for layer_idx in range(num_layers):
        # Access the i-th layer
        layer = model.model.layers[layer_idx].self_attn
        
        # Register forward hooks
        hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
        hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
        
        hooks_k.append(hook_k)
        hooks_v.append(hook_v)

    return model, hooks_k, hooks_v, collected_k_outputs, collected_v_outputs

from datasets import load_dataset
DATADIR = {
    'ruler': '/data/shenth/datasets/ruler'
}

class Dataset:
    def __init__(self, dataset_name, tokenizer, datalen, num_samples, rank=0, world_size=1):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.datalen = datalen
        self.num_samples = num_samples
        self.rank = rank
        self.world_size = world_size
        self.is_sharded = False

        if dataset_name == 'niah':
            self.tokenized_prompts, self.gt, self.ctx_len, self.depth_pct = self.get_dataset()
        else:
            self.tokenized_prompts, self.gt = self.get_dataset()
        
        self.num_samples = len(self.tokenized_prompts)
        self.gen_len = self.get_gen_len()

    def __str__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __repr__(self) -> str:
        return f"Dataset: {self.dataset_name}, Num Samples: {self.num_samples}, Gen Len: {self.gen_len}, DataLen: {self.datalen}"

    def __len__(self) -> int:
        return self.num_samples

    def shard(self, rank, world_size):
        if world_size > 1:
            shard_size = self.num_samples // world_size
            start = rank * shard_size
            end = start + shard_size if rank != world_size - 1 else self.num_samples
            shard_tokenized_prompts, shard_gt = self.tokenized_prompts[start:end], self.gt[start:end]
            self.tokenized_prompts = shard_tokenized_prompts
            self.gt = shard_gt
            self.num_samples = len(shard_tokenized_prompts)

        self.is_sharded = True

    def get_gen_len(self):
        if 'niah' == self.dataset_name:
            return 10
        elif 'niah' in self.dataset_name:
            return 128
        elif 'vt' in self.dataset_name:
            return 30
        elif 'cwe' in self.dataset_name:
            return 120
        elif 'fwe' in self.dataset_name:
            return 50
        elif 'qa' in self.dataset_name:
            return 32
        else:
            raise Exception("Gen len not found")

    def __getitem__(self, idx):
        if 'persona' in self.dataset_name:
            return self.tokenized_prompts[idx], self.queries[idx], self.gt[idx]
        return self.tokenized_prompts[idx], self.gt[idx]

    def get_dataset(self):
        if 'ruler' in self.dataset_name: # ruler/xxx
            task = self.dataset_name.split('/')[-1]

            dataset = load_dataset(path=f'{DATADIR["ruler"]}', split="test")
            if self.num_samples > 0:
                self.num_samples = min(self.num_samples, len(dataset))
            else:
                self.num_samples = len(dataset)
            tokenized_prompts = []
            gt = []

            sample = 0
            for i in range(len(dataset)):
                if sample == self.num_samples: break
                if dataset[i].get('task') == task:
                    input_text = dataset[i]['context']
                    #input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False)
                    input_ids = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False)
                    tokenized_prompts.append(input_ids)
                    gt.append(dataset[i]['answer'])
                    sample = sample + 1
                

            return tokenized_prompts, gt
        else:
            raise ValueError(f"Dataset {self.dataset_name} not found, please choose in ruler, persona, infini_bench, needle, niah, long_bench")

dataset = Dataset("ruler/niah_multivalue", tokenizer, 4096, 4, -1, 1)

model, hooks_k, hooks_v, collected_k_outputs, collected_v_outputs = load_and_process_model()
num_layers = len(model.model.layers)
with torch.no_grad():
    for i in range(dataset.num_samples):
        print(f"Processing {i}")
        prompt = dataset.tokenized_prompts[i]
        input_ids = prompt["input_ids"].cuda()
        attention_mask = prompt["attention_mask"].cuda()
        model(input_ids, attention_mask=attention_mask)
        break


for hook in hooks_k:
    hook.remove()
for hook in hooks_v:
    hook.remove()

print("Num samles (layers) collected:", len(collected_k_outputs))

import torch
import torch.nn.functional as F

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

import matplotlib.pyplot as plt
import numpy as np

def plot_tensor(w, ax=None, title=None, x_label="Dims", y_label="Token"):
    # Prepare the data for plotting
    x = np.arange(w.shape[0])  # Dims
    y = np.arange(w.shape[1])  # Seqlen
    x, y = np.meshgrid(x, y)

    # Use the provided axis or create a new figure and axis
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None

    # Plot the surface
    surf = ax.plot_surface(x, y, w.T, cmap='coolwarm')
    ax.xaxis.set_tick_params(pad=-6, size=13)
    ax.yaxis.set_tick_params(pad=-4, size=13)
    ax.zaxis.set_tick_params(pad=-3, size=13)
    
    # Set title and labels
    if title is not None:
        ax.set_title(title, fontsize=15)
    ax.set_ylabel(x_label, labelpad=-5, fontsize=14)
    ax.set_xlabel(y_label, labelpad=-1, fontsize=14)
    ax.set_zlabel('Absolute Activation Value', labelpad=-5, fontsize=14)

    return fig

def plot_vector(w, ax=None, title=None, x_label="Dims", y_label="Value"):
    # Use the provided axis or create a new figure and axis
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
    else:
        fig = None

    # Plot the vector
    ax.plot(w)
    ax.xaxis.set_tick_params(size=16)
    ax.yaxis.set_tick_params(size=16)
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=14)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=14)
    # Set title
    if title is not None:
        ax.set_title(title)

    return fig


def plot_side_by_side(tensors, titles=None, plot_type='tensor', saved_path = None):
    """
    Plot multiple tensors or vectors side by side.

    Args:
        tensors: List of tensors or vectors to plot.
        titles: List of titles for each plot (optional).
        plot_type: Type of plot ('tensor' for 3D tensor, 'vector' for 1D vector).
    """
    num_plots = len(tensors)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots + 3, 6), subplot_kw={'projection': '3d'} if plot_type == 'tensor' else {})

    # In case of only one plot, axs won't be an array; make it an array for consistency
    if num_plots == 1:
        axs = [axs]

    # Plot each tensor/vector
    for i, (tensor, ax) in enumerate(zip(tensors, axs)):
        title = titles[i] if titles is not None else None
        if plot_type == 'tensor':
            plot_tensor(tensor, ax=ax, title=title)
        elif plot_type == 'vector':
            plot_vector(tensor, ax=ax, title=title)
        else:
            raise ValueError("Unsupported plot type. Use 'tensor' or 'vector'.")

    plt.tight_layout()
    if saved_path is not None:
        plt.savefig(saved_path)
    plt.show()

import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

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
mode = "Key"
if mode == "Key":
    for i in range(num_layers):
        for j in range(num_layers):
            ki = collected_k_outputs[i]
            kj = collected_k_outputs[j]
            
            assert ki.shape == kj.shape
            assert ki.shape[0] == 1 # batch size is 1
            
            ki = ki.squeeze(0).cuda().float()
            kj = kj.squeeze(0).cuda().float()
            cka_matrix[i, j] = linear_cka_centered_torch(ki, kj)
            print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")
            
            del ki, kj
elif mode == "Value":
    for i in range(num_layers):
        for j in range(num_layers):
            vi = collected_v_outputs[i]
            vj = collected_v_outputs[j]
            
            assert vi.shape == vj.shape
            assert vi.shape[0] == 1

            vi = vi.squeeze(0).cuda().float()
            vj = vj.squeeze(0).cuda().float()
            cka_matrix[i, j] = linear_cka_centered_torch(vi, vj)
            print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")
elif mode == "KV":
    for i in range(num_layers):
        for j in range(num_layers):
            ki = collected_k_outputs[i]
            vj = collected_v_outputs[j]
            
            assert not torch.all(ki == vj)
            
            assert ki.shape == vj.shape
            assert ki.shape[0] == 1
            
            ki = ki.squeeze(0).cuda().float()
            vj = vj.squeeze(0).cuda().float()
            cka_matrix[i, j] = linear_cka_centered_torch(ki, vj)
            print(f"CKA({i}, {j}) = {cka_matrix[i, j]}")


cmap = sns.diverging_palette(240, 10)
custom_colors = ["#D93F49", "#E28187", "#EBBFC2", "#D5E1E3", "#AFC9CF", "#8FB4BE"]  
custom_colors = custom_colors[::-1]  # Reverse the colors for better contrast
plot_heatmap(cka_matrix, title=f"CKA Matrix ({mode}-Cache)", colorbar=True, x_label="Layer", y_label="Layer", custom_colors=custom_colors)