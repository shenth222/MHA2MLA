import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from IPython import embed
from loguru import logger
import torch_dct as dct

model_name = "/data/shenth/models/llama/2-7b-hf"  # Or any other Llama checkpoint available
tokenizer = AutoTokenizer.from_pretrained(model_name)
def load_and_process_model(collect_q=False, collect_k=False, collect_v=False):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()
    model = model.cuda()

    # We'll collect outputs for all layers in these lists
    collected_k_outputs = []
    collected_v_outputs = []
    collected_q_outputs = []
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

    def q_proj_hook(module, input, output):
        """
        Same logic as k_proj_hook, but for q_proj.
        """
        collected_q_outputs.append(output.detach().cpu())


    num_layers = len(model.model.layers)
    hooks_k = []
    hooks_v = []
    hooks_q = []
    for layer_idx in range(num_layers):
        # Access the i-th layer
        layer = model.model.layers[layer_idx].self_attn
        
        # Register forward hooks
        if collect_k:
            hook_k = layer.k_proj.register_forward_hook(k_proj_hook)
            hooks_k.append(hook_k)
        if collect_v:
            hook_v = layer.v_proj.register_forward_hook(v_proj_hook)
            hooks_v.append(hook_v)
        if collect_q:
            hook_q = layer.q_proj.register_forward_hook(q_proj_hook)
            hooks_q.append(hook_q)

    return model, (hooks_q, hooks_k, hooks_v), (collected_q_outputs, collected_k_outputs, collected_v_outputs)

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

model, hooks, collected_outputs = load_and_process_model(False, True, False)
num_layers = len(model.model.layers)

with torch.no_grad():
    for i in range(dataset.num_samples):
        print(f"Processing {i}")
        prompt = dataset.tokenized_prompts[i]
        input_ids = prompt["input_ids"].cuda()
        attention_mask = prompt["attention_mask"].cuda()
        model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=128)
        break


for hook in hooks[1]:
    hook.remove()
for hook in hooks[2]:
    hook.remove()
for hook in hooks[0]:
    hook.remove()
# print("Num samles (layers) collected:", len(collected_outputs[1]))

import torch
import torch.nn.functional as F
import torch_dct as dct
import matplotlib.pyplot as plt
import seaborn as sns
mode = "Key"
if mode == "Key":
    collected_k_outputs = collected_outputs[1]
    if not collected_k_outputs:
        logger.error("CKA mode is {}, but {} is None".format(mode, mode))

    ki = collected_k_outputs[26]
    ki = ki.squeeze(0).cpu().float()
    output = torch.abs(dct.dct_2d(ki))

    plt.figure(figsize=(7, 6))
    plt.title("DCT result of llama2-7b layer 27 K cache")
    im = plt.imshow(output, cmap='cool', aspect='auto')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=19)
    plt.savefig("./figure/k_cache_dct_res.png",)
    plt.close()

    # 绘制分布频率曲线图
    plt.figure(figsize=(8, 6))
    sns.kdeplot(output.cpu().numpy().flatten(), color='blue', linewidth=2, label="Frequency Curve")
    plt.title("Frequency Distribution Curve of DCT result Values of llama2-7b layer 27 K cache", fontsize=16)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.legend(fontsize=12)
    plt.savefig("./figure/k_cache_frequency_curve.png")
    plt.close()
