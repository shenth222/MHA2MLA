import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from IPython import embed
from loguru import logger
import torch_dct as dct

model_name = "/data/shenth/models/llama/2-7b-hf"  # Or any other Llama checkpoint available

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model = model.eval()

input = model.model.layers[26].self_attn.k_proj.weight.detach()
output = torch.abs(dct.dct_2d(input))
print(output.size())

import plotly.express as px
import matplotlib.pyplot as plt
import os
# fig = px.imshow(
#     output,
#     labels=dict(x='dimension', y='dimension', color="value"),
#     x=list(range(4096)),
#     y=list(range(4096)),
#     color_continuous_scale='gray'
# )
# os.makedirs("./figure", exist_ok=True)
# fig.write_image("./figure/test.png")

os.makedirs("./figure", exist_ok=True)

plt.figure(figsize=(7, 6))
plt.title("DCT result of llama2-7b layer 27 W_k weights")
im = plt.imshow(output, cmap='cool', aspect='auto')
cbar = plt.colorbar(im)
cbar.ax.tick_params(labelsize=19)
plt.savefig("./figure/dct_res.png",)
plt.close()

plt.figure(figsize=(8, 6))
plt.hist(output.cpu().numpy().flatten(), bins=50, color='blue', alpha=0.7)
plt.title("Frequency Distribution of DCT result Values of llama2-7b layer 27 W_k weights", fontsize=16)
plt.xlabel("Value", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(axis='y', alpha=0.75)

# 保存频率分布图
plt.savefig("./figure/frequency_distribution.png")
plt.close()