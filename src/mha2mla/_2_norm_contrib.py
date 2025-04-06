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

model_name = "135m" # "135m", "2-7b", "360m"
path = "/data/shenth/models/SmolLM/135m"

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
    ################## one sample ##################
    def random_one_sample(find_the_longest=False):
        dataloader_list = list(test_dataloader)
        if find_the_longest:
            batch = max(dataloader_list, key=lambda x: torch.sum(x["attention_mask"]).item())
        else:
            batch = random.choice(dataloader_list)
        batch = {k: v.to("cuda") for k, v in batch.items()}
        clear_qk_states()
        q, _ = get_qk_states()
        if q is not None:
            AssertionError("q should be None")
        with torch.no_grad():
            model(**batch)
            q, k = get_qk_embed_states() # [layer_idx, bsz, num_heads, seq_len, head_dim]
        assert q, "sth wrong with q & k"

        q = torch.stack(q, dim=0)  # [layer_idx, bsz, num_heads, seq_len, head_dim]
        k = torch.stack(k, dim=0)  

        num_ones = torch.sum(batch["attention_mask"])
        q = q[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]
        k = k[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]

        attn_weights = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(config.head_dim)
        attn_weights = attn_weights.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

        def cal_2_norm(states):
            # [layer_idx, bsz, num_heads, seq_len, head_dim]
            layer_idx, bsz, num_heads, seq_len, head_dim = states.shape
            states = states.reshape(layer_idx, bsz, num_heads, seq_len, 2, -1).transpose(-1, -2)
            # [layer_idx, bsz, num_heads, seq_len, head_dim//2, 2]
            states = states.norm(dim=-1, p=2)
            # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
            return states

        q_norm = cal_2_norm(q) # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
        k_norm = cal_2_norm(k)
        attn_weights_norm = torch.matmul(q_norm, k_norm.transpose(3, 4)) / math.sqrt(config.head_dim)
        attn_weights_norm = attn_weights_norm.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

        def make_heatmap(
                states,
                flag, 
                model_name,
                save_dir,
                x_label=None,
                y_label=None,
                title=None,
                color_scale="YlGnBu",
                aspect="auto",
                save_name=None,
        ):  
            # [layer_idx, num_heads, _, _]

            if save_dir is None:
                AssertionError("save_dir should not be None")
            if model_name not in ["135m", "2-7b", "360m"]:
                raise ValueError("model_name should be in [135m, 2-7b, 360m]")
            
            for layer_idx in range(config.num_hidden_layers):
                for num_heads in range(config.num_attention_heads):
                    data = states[layer_idx][num_heads].cpu().numpy()
                    fig = px.imshow(
                        data,
                        labels=dict(x=x_label, y=y_label, color="Value"),
                        x=list(range(data.shape[1])),
                        y=list(range(data.shape[0])),
                        title=title,
                        color_continuous_scale=color_scale,
                        aspect=aspect,
                    )

                    os.makedirs(save_dir, exist_ok=True)
                    fig.write_image(f"{save_dir}/{save_name}.png")

        # make_heatmap(attn_weights, "attn_weights", model_name, f"./figure/norm_contrib/{model_name}/attn_weights/")
        # make_heatmap(attn_weights_norm, "attn_weights_norm", model_name, f"./figure/norm_contrib/{model_name}/attn_weights_norm/")

        def real_contrib(q, k):
            # layer_idx, bsz, num_heads, seq_len, head_dim
            q = q.mean(dim=3) # [layer_idx, bsz, num_heads, head_dim]
            k = k.mean(dim=3)
            q = q.mean(dim=1) # [layer_idx, num_heads, head_dim]
            k = k.mean(dim=1)

            q = q.unsqueeze(2) # [layer_idx, num_heads, 1, head_dim]
            k = k.unsqueeze(2)

            attn_res_baseline = torch.matmul(q, k.transpose(2, 3)) # [layer_idx, num_heads, 1, 1]
            head_dim = q.shape[-1]
            contribution = []
            for idx in range(head_dim):
                q_tmp = q.clone()
                k_tmp = k.clone()
                q_tmp[:, :, :, idx] = 0
                k_tmp[:, :, :, idx] = 0
                attn_res_modified = torch.matmul(q_tmp, k_tmp.transpose(2, 3))
                contrib = torch.abs((attn_res_baseline - attn_res_modified) / attn_res_baseline) * 100 # [layer_idx, num_heads, 1, 1]
                contrib = contrib.squeeze(-1).squeeze(-1)
                contribution.append(contrib)
            contribution = torch.stack(contribution, dim=-1) # [layer_idx, num_heads, head_dim]
            return contribution

        def norm_contrib(q, k):
            # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
            q = q.mean(dim=3) # [layer_idx, bsz, num_heads, head_dim//2]
            k = k.mean(dim=3)
            q = q.mean(dim=1) # [layer_idx, num_heads, head_dim//2]
            k = k.mean(dim=1)
            res = q * k # [layer_idx, num_heads, head_dim//2]
            res = torch.repeat_interleave(res, 2, dim=-1) # [layer_idx, num_heads, head_dim]
            return res
        
        real_contribution = real_contrib(q, k) # [layer_idx, num_heads, head_dim]
        norm_contribution = norm_contrib(q_norm, k_norm) # [layer_idx, num_heads, head_dim]
        
        def contrib_heatmap(
                contribution,
                flag,
                model_name,
                save_dir,
                x_label=None,
                y_label=None,
                color_scale="YlGnBu",
                aspect="auto",
        ):
            if save_dir is None:
                AssertionError("save_dir should not be None")
            if model_name not in ["135m", "2-7b", "360m"]:
                raise ValueError("model_name should be in [135m, 2-7b, 360m]")

            # layer_idx, num_heads, head_dim
            for layer_idx in range(config.num_hidden_layers):
                data = contribution[layer_idx].transpose(0,1).cpu().numpy()
                fig = px.imshow(
                    data,
                    labels=dict(x=x_label, y=y_label, color="Value"),
                    x=list(range(data.shape[1])),
                    y=list(range(data.shape[0])),
                    title=f"Layer {layer_idx} {flag}",
                    color_continuous_scale=color_scale,
                    aspect=aspect,
                )
                os.makedirs(save_dir, exist_ok=True)
                fig.write_image(f"{save_dir}/Layer_{layer_idx}_{flag}.png")

        contrib_heatmap(
            real_contribution,
            "real_contribution",
            model_name,
            f"./figure/norm_contrib/{model_name}/real_contribution/",
            x_label="num_heads",
            y_label="head_dim",
        )

        contrib_heatmap(
            norm_contribution,
            "norm_contribution",
            model_name,
            f"./figure/norm_contrib/{model_name}/norm_contribution/",
            x_label="num_heads",
            y_label="head_dim",
        )

    random_one_sample(find_the_longest=True)


    