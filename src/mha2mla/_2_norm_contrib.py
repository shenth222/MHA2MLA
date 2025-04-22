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
import pandas as pd
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

def cal_2_norm(states):
    # [layer_idx, bsz, num_heads, seq_len, head_dim]
    layer_idx, bsz, num_heads, seq_len, head_dim = states.shape
    states = states.reshape(layer_idx, bsz, num_heads, seq_len, 2, -1).transpose(-1, -2)
    # [layer_idx, bsz, num_heads, seq_len, head_dim//2, 2]
    states = states.norm(dim=-1, p=2)
    # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
    return states

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
        config=None,
):  
    # [layer_idx, num_heads, _, _]
    if config is None:
        AssertionError("config should not be None")
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
    total_diff = []
    for idx in range(head_dim):
        q_tmp = q.clone()
        k_tmp = k.clone()
        q_tmp[:, :, :, idx] = 0
        k_tmp[:, :, :, idx] = 0
        attn_res_modified = torch.matmul(q_tmp, k_tmp.transpose(2, 3))
        ######## normalized ########
        diff = torch.abs(attn_res_baseline - attn_res_modified).squeeze(-1).squeeze(-1) # [layer_idx, num_heads]
        total_diff.append(diff)
    contribution = torch.stack(total_diff, dim=-1) # [layer_idx, num_heads, head_dim]
    total_diff = torch.stack(total_diff, dim=-1) # [layer_idx, num_heads, head_dim]
    total_diff = torch.sum(total_diff, dim=-1) # [layer_idx, num_heads]
    contribution = contribution / total_diff.unsqueeze(-1) # [layer_idx, num_heads, head_dim]

        ######## unnormalized ########
        # contrib = torch.abs((attn_res_baseline - attn_res_modified) / attn_res_baseline) * 100 # [layer_idx, num_heads, 1, 1]
        # contrib = contrib.squeeze(-1).squeeze(-1)
        # contribution.append(contrib)
    # contribution = torch.stack(contribution, dim=-1) # [layer_idx, num_heads, head_dim]
    return contribution

def norm_contrib(q, k):
    # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
    q = q.mean(dim=3) # [layer_idx, bsz, num_heads, head_dim//2]
    k = k.mean(dim=3)
    q = q.mean(dim=1) # [layer_idx, num_heads, head_dim//2]
    k = k.mean(dim=1)
    res = q * k # [layer_idx, num_heads, head_dim//2]
    res = torch.concat((res,res), dim=-1) # [layer_idx, num_heads, head_dim]
    return res

def get_top_r_mask(norm_contrib, topr=4):
    topr = topr * 2
    top_indices = torch.topk(norm_contrib, topr, dim=-1).indices # [layer_idx, num_heads, topr]
    mask = torch.zeros_like(norm_contrib, dtype=torch.bool) # [layer_idx, num_heads, head_dim]
    mask.scatter_(-1, top_indices, True) # [layer_idx, num_heads, head_dim]
    return mask

def contrib_heatmap(
        contribution,
        flag,
        model_name,
        save_dir,
        x_label=None,
        y_label=None,
        color_scale="YlGnBu",
        aspect="auto",
        config=None,
):
    if config is None:
        AssertionError("config should not be None")
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

def calculate_recovery_rate(real_contrib, mask):
    real_contrib = real_contrib * mask # [layer_idx, num_heads, head_dim]
    recovery_rate = real_contrib.sum(dim=-1) # [layer_idx, num_heads]
    return recovery_rate

if LOAD_STATES:
    # need fix bugs
    pass
else:
    config = load_config(path)
    tokenizer = load_tokenizer(path)
    model = load_model(path, config)
    # datasets
    data_path = "/data/shenth/datasets/glue"
    test_dataloader = make_datasets(data_path, tokenizer, bsz=8)
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

        attn_weights = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(config.head_dim) # [layer_idx, bsz, num_heads, seq_len, seq_len]
        attn_weights = attn_weights.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

        q_norm = cal_2_norm(q) # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
        k_norm = cal_2_norm(k)
        attn_weights_norm = torch.matmul(q_norm, k_norm.transpose(3, 4)) / math.sqrt(config.head_dim)
        attn_weights_norm = attn_weights_norm.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

        # make_heatmap(attn_weights, "attn_weights", model_name, f"./figure/norm_contrib/{model_name}/attn_weights/")
        # make_heatmap(attn_weights_norm, "attn_weights_norm", model_name, f"./figure/norm_contrib/{model_name}/attn_weights_norm/")
        
        real_contribution = real_contrib(q, k) # [layer_idx, num_heads, head_dim]
        norm_contribution = norm_contrib(q_norm, k_norm) # [layer_idx, num_heads, head_dim]
        
        ############ attn_res recovery ############
        THRESHOLD = 0.9
        top_r = 4

        mask = get_top_r_mask(norm_contribution, top_r) # [layer_idx, num_heads, head_dim]
        
        recovery_rate = calculate_recovery_rate(real_contribution, mask) # [layer_idx, num_heads]
        
        ######### boxplot #########
        # data = {
        #     "Layer": [],
        #     "Recovery Rate": [],
        # }
        # for layer_idx in range(config.num_hidden_layers):
        #     for num_heads in range(config.num_attention_heads):
        #         data["Layer"].append(layer_idx)
        #         data["Recovery Rate"].append(recovery_rate[layer_idx, num_heads].item())
                
        # df = pd.DataFrame(data)
        # fig = px.box(
        #     df,
        #     x="Layer",
        #     y="Recovery Rate",
        #     title=f"Recovery Rate by Layer of {model_name}",
        #     labels={"Layer": "Layer Index", "Recovery Rate": "Recovery Rate"},
        #     template="plotly_white",
        #     color_discrete_sequence=px.colors.qualitative.Light24[0:config.num_hidden_layers],
        #     color="Layer",
        # )
        # attn_recovery_rate_dir = f"./figure/attn_recovery_rate/{model_name}/recovery_rate/"
        # os.makedirs(attn_recovery_rate_dir, exist_ok=True)
        # fig.write_image(f"{attn_recovery_rate_dir}recovery_rate.png", scale=3)

        contrib_heatmap(
            real_contribution,
            "real_contribution",
            model_name,
            f"./figure/norm_contrib/{model_name}/real_contribution/",
            x_label="num_heads",
            y_label="head_dim",
            config = config
        )

        contrib_heatmap(
            norm_contribution,
            "norm_contribution",
            model_name,
            f"./figure/norm_contrib/{model_name}/norm_contribution/",
            x_label="num_heads",
            y_label="head_dim",
            config = config
        )

    random_one_sample(find_the_longest=True)

    def random_n_samples(n=None):
        recovery_rate_list = []
        if n is not None and n < len(test_dataloader):
            dataloader_list = random.sample(list(test_dataloader), n)
            dataloader_list = list(test_dataloader)
            for batch in tqdm(dataloader_list, desc="Processing batches", total=n):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                clear_qk_states()
                q, _ = get_qk_states()
                if q is not None:
                    AssertionError("sth wrong with q & k")
                with torch.no_grad():
                    model(**batch)
                    q, k = get_qk_embed_states() # [layer_idx, bsz, num_heads, seq_len, head_dim]
                assert q, "sth wrong with q & k"

                q = torch.stack(q, dim=0)  # [layer_idx, bsz, num_heads, seq_len, head_dim]
                k = torch.stack(k, dim=0)  

                num_ones = torch.sum(batch["attention_mask"])
                q = q[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]
                k = k[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]

                attn_weights = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(config.head_dim) # [layer_idx, bsz, num_heads, seq_len, seq_len]
                attn_weights = attn_weights.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

                q_norm = cal_2_norm(q) # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
                k_norm = cal_2_norm(k)
                attn_weights_norm = torch.matmul(q_norm, k_norm.transpose(3, 4)) / math.sqrt(config.head_dim)
                attn_weights_norm = attn_weights_norm.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

                real_contribution = real_contrib(q, k) # [layer_idx, num_heads, head_dim]
                norm_contribution = norm_contrib(q_norm, k_norm) # [layer_idx, num_heads, head_dim]
                
                ############ attn_res recovery ############
                THRESHOLD = 0.9
                top_r = 4

                mask = get_top_r_mask(norm_contribution, top_r) # [layer_idx, num_heads, head_dim]
                
                recovery_rate = calculate_recovery_rate(real_contribution, mask) # [layer_idx, num_heads]
                recovery_rate_list.append(recovery_rate)
        else:
            for batch in tqdm(test_dataloader, desc="Processing batches", total=len(test_dataloader)):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                clear_qk_states()
                q, _ = get_qk_states()
                if q is not None:
                    AssertionError("sth wrong with q & k")
                with torch.no_grad():
                    model(**batch)
                    q, k = get_qk_embed_states() # [layer_idx, bsz, num_heads, seq_len, head_dim]
                assert q, "sth wrong with q & k"

                q = torch.stack(q, dim=0)  # [layer_idx, bsz, num_heads, seq_len, head_dim]
                k = torch.stack(k, dim=0)  

                num_ones = torch.sum(batch["attention_mask"])
                q = q[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]
                k = k[:, :, :, :num_ones, :]  # [layer_idx, bsz, num_heads, seq_len, head_dim]

                attn_weights = torch.matmul(q, k.transpose(3, 4)) / math.sqrt(config.head_dim) # [layer_idx, bsz, num_heads, seq_len, seq_len]
                attn_weights = attn_weights.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

                q_norm = cal_2_norm(q) # [layer_idx, bsz, num_heads, seq_len, head_dim//2]
                k_norm = cal_2_norm(k)
                attn_weights_norm = torch.matmul(q_norm, k_norm.transpose(3, 4)) / math.sqrt(config.head_dim)
                attn_weights_norm = attn_weights_norm.mean(dim=1) # [layer_idx, num_heads, seq_len, seq_len]

                real_contribution = real_contrib(q, k) # [layer_idx, num_heads, head_dim]
                norm_contribution = norm_contrib(q_norm, k_norm) # [layer_idx, num_heads, head_dim]
                
                ############ attn_res recovery ############
                THRESHOLD = 0.9
                top_r = 4

                mask = get_top_r_mask(norm_contribution, top_r) # [layer_idx, num_heads, head_dim]
                
                recovery_rate = calculate_recovery_rate(real_contribution, mask) # [layer_idx, num_heads]
                recovery_rate_list.append(recovery_rate)

        recovery_rate = torch.stack(recovery_rate_list, dim=0) # [total_sample/bsz, layer_idx, num_heads]
        recovery_rate = recovery_rate.mean(dim=0) # [layer_idx, num_heads]
            
        ######### boxplot #########
        data = {
            "Layer": [],
            "Recovery Rate": [],
        }
        for layer_idx in range(config.num_hidden_layers):
            for num_heads in range(config.num_attention_heads):
                data["Layer"].append(layer_idx)
                data["Recovery Rate"].append(recovery_rate[layer_idx, num_heads].item())
                
        df = pd.DataFrame(data)
        fig = px.box(
            df,
            x="Layer",
            y="Recovery Rate",
            title=f"Recovery Rate by Layer of {model_name}",
            labels={"Layer": "Layer Index", "Recovery Rate": "Recovery Rate"},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Light24[0:config.num_hidden_layers],
            color="Layer",
        )
        attn_recovery_rate_dir = f"./figure/attn_recovery_rate/{model_name}/recovery_rate/"
        os.makedirs(attn_recovery_rate_dir, exist_ok=True)
        fig.write_image(f"{attn_recovery_rate_dir}recovery_rate_glue.png", scale=3)
    
    # random_n_samples()