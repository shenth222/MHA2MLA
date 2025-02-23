"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""

import argparse
import os
import yaml
from typing import Dict, cast
from types import MethodType
import seaborn as sns
import torch

import numpy as np
import matplotlib.pyplot as plt
from nanotron import logging
from nanotron.config import (
    DataArgs,
    DatasetStageArgs,
    NanosetDatasetsArgs,
    PretrainDatasetsArgs,
)
from nanotron.data.dataloader_builder import build_nanoset_dataloader
from nanotron.dataloader import (
    clm_process,
    dummy_infinite_data_generator,
    get_datasets,
    get_train_dataloader,
)
from nanotron.helpers import (
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
)
from nanotron.logging import log_rank
from nanotron.parallel.pipeline_parallel.utils import get_input_output_pp_ranks
from nanotron.trainer import DistributedTrainer
from nanotron.utils import main_rank_first
from torch.utils.data import DataLoader

try:
    from huggingface_hub import __version__ as hf_hub_version
    from transformers import AutoTokenizer
    from transformers import __version__ as tf_version
except ImportError:
    hf_hub_version = None
    tf_version = None

logger = logging.get_logger(__name__)


def get_dataloader_from_data_stage(
    trainer: DistributedTrainer,
    data: DataArgs,
    consumed_train_samples: int,
    num_remaining_train_steps: int,
):
    """
    Returns a dataloader for a given data stage.

    data: The data configuration for the current stage.
    consumed_train_samples: The number of samples consumed by the model in the this stage (each stage starts from zero).
    num_remaining_train_steps: The number of remaining training steps for this stage.
    """
    assert (
        consumed_train_samples >= 0
    ), "consumed_train_samples should be greater than 0"
    assert (
        num_remaining_train_steps >= 0
    ), "num_remaining_train_steps should be greater than 0"

    # First, we need to know which ranks to feed the dataloader to
    input_pp_rank, output_pp_rank = get_input_output_pp_ranks(model=trainer.model)

    # Case 1: Dummy data generator
    if data.dataset is None:
        log_rank(
            "Using dummy data generator", logger=logger, level=logging.INFO, rank=0
        )
        dataloader = dummy_infinite_data_generator(
            micro_batch_size=trainer.micro_batch_size,
            sequence_length=trainer.sequence_length,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            vocab_size=trainer.model_config.vocab_size,
            seed=data.seed,
            parallel_context=trainer.parallel_context,
        )()

    # Case 2: HuggingFace datasets
    elif isinstance(data.dataset, PretrainDatasetsArgs):
        log_rank("Using `datasets` library", logger=logger, level=logging.INFO, rank=0)
        tokenizer_path = trainer.config.tokenizer.tokenizer_name_or_path
        log_rank(
            f"Loading tokenizer from {tokenizer_path} and transformers/hf_hub versions {tf_version, hf_hub_version}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        # We need to the 1st device to process dataset and cache it, then other devices load from cache
        with main_rank_first(trainer.parallel_context.world_pg):
            # TODO @nouamanetazi: this may timeout before 1st device finishes processing dataset. Can we have a ctxmanager to modify timeout?
            # TODO: generalise to include  for validation/test splits

            # We load the raw dataset
            raw_dataset = get_datasets(
                hf_dataset_or_datasets=data.dataset.hf_dataset_or_datasets,
                hf_dataset_config_name=data.dataset.hf_dataset_config_name,
                splits=data.dataset.hf_dataset_splits,
            )["train"]

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            # Check that tokenizer's vocab size is smaller than the model's vocab size
            assert (
                tokenizer.vocab_size <= trainer.model_config.vocab_size
            ), f"Tokenizer's vocab size ({tokenizer.vocab_size}) is larger than the model's vocab size ({trainer.model_config.vocab_size})"

            # We apply the Causal Language Modeling preprocessing
            train_dataset = clm_process(
                raw_dataset=raw_dataset,
                tokenizer=tokenizer,
                text_column_name=data.dataset.text_column_name,
                dataset_processing_num_proc_per_process=data.dataset.dataset_processing_num_proc_per_process,
                dataset_overwrite_cache=data.dataset.dataset_overwrite_cache,
                sequence_length=trainer.sequence_length,
            )

            # We load the processed dataset on the ranks requiring it
            dataloader = get_train_dataloader(
                train_dataset=train_dataset,
                sequence_length=trainer.sequence_length,
                parallel_context=trainer.parallel_context,
                input_pp_rank=input_pp_rank,
                output_pp_rank=output_pp_rank,
                micro_batch_size=trainer.micro_batch_size,
                consumed_train_samples=consumed_train_samples,
                dataloader_num_workers=data.num_loading_workers,
                seed_worker=data.seed,
                dataloader_drop_last=True,
            )

            # Check if we have enough samples for train_steps
            total_tokens_dataset = len(dataloader.dataset) * trainer.sequence_length
            num_tokens_needed_for_training = (
                num_remaining_train_steps
                * trainer.global_batch_size
                * trainer.sequence_length
            )
            assert num_tokens_needed_for_training <= total_tokens_dataset, (
                f"Dataset is too small for steps ({total_tokens_dataset} < {num_tokens_needed_for_training}), "
                f"Try train_steps<={len(dataloader.dataset) // trainer.global_batch_size + trainer.iteration_step}"
            )

    # Case 3: Nanosets
    elif isinstance(data.dataset, NanosetDatasetsArgs):
        # Get tokenizer cardinality
        tokenizer = AutoTokenizer.from_pretrained(
            trainer.config.tokenizer.tokenizer_name_or_path
        )
        token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
        del tokenizer
        # Create Nanoset
        from nanotron.data.nanoset import Nanoset

        with main_rank_first(trainer.parallel_context.world_pg):
            train_dataset = Nanoset(
                dataset_folders=data.dataset.dataset_folder,
                dataset_weights=data.dataset.dataset_weights,
                sequence_length=trainer.sequence_length,
                token_size=token_size,
                train_split_num_samples=trainer.config.tokens.train_steps
                * trainer.global_batch_size,
                random_seed=data.seed,
            )

        # Prepare dataloader
        train_dataloader = build_nanoset_dataloader(
            train_dataset,
            trainer.sequence_length,
            parallel_context=trainer.parallel_context,
            input_pp_rank=input_pp_rank,
            output_pp_rank=output_pp_rank,
            micro_batch_size=trainer.micro_batch_size,
            consumed_train_samples=consumed_train_samples,
            dataloader_num_workers=data.num_loading_workers,
            dataloader_drop_last=True,
        )

        return train_dataloader
    else:
        raise ValueError(
            f"Unhandled case of `self.config.data.dataset`. Got: {data.dataset}"
        )

    return dataloader


def get_dataloader(trainer: DistributedTrainer) -> Dict[str, DataLoader]:
    dataloaders = {}

    for stage_idx, stage in enumerate(trainer.config.data_stages):
        # NOTE: we only create the dataloader for the first stage,
        # then we lazy initialize the dataloader for the other stages
        stage = cast(DatasetStageArgs, stage)
        consumed_train_samples = get_consumed_train_samples_of_a_data_stage_from_ckp(
            stage, trainer.metadata
        )
        assert (
            consumed_train_samples is not None
        ), f"Cannot find consumed_train_samples for stage {stage.start_training_step} in the checkpoint"

        num_remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
            stage, trainer.config, trainer.metadata
        )
        log_rank(
            f"[Training Plan] Stage {stage.name} has {num_remaining_train_steps} remaining training steps and has consumed {consumed_train_samples} samples",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        dataloader = (
            get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
            if stage_idx == 0
            else lambda stage=stage: get_dataloader_from_data_stage(
                trainer,
                stage.data,
                consumed_train_samples=consumed_train_samples,
                num_remaining_train_steps=num_remaining_train_steps,
            )
        )
        dataloaders[stage.name] = dataloader
    return dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the YAML or python config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1024,
    )
    return parser.parse_args()


query_states_dict = {}
key_states_dict = {}

from typing import Optional, Tuple, Union


def get_rotary_forward(module_name):
    def my_rotary_forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        global query_states_dict
        global key_states_dict
        query_states_dict[module_name] = qkv
        key_states_dict[module_name] = torch.split(kv, 1, dim=2)[0]
        return self._forward(qkv, kv, seqlen_offset, max_seqlen, num_heads_q)

    return my_rotary_forward


def flatten_avg_query_key_states(query_states: dict, key_states: dict):
    max_num_heads = max(
        list(query_states_dict.values())[0].shape[2],
        list(key_states_dict.values())[0].shape[2],
    )

    def flatten_dict(states: dict):
        f_states = []
        for k, v in states.items():  # states: {module_name: hidden_states}
            item = []
            v = v.squeeze()  # bsz,seqlen,head,head_dim
            if len(v.shape) == 3:
                v = v.unsqueeze(0)
            v = torch.norm(
                v.reshape(v.shape[0], v.shape[1], v.shape[2], 2, -1).transpose(-1, -2),
                p=2,
                dim=4,
            )
            block_size = max_num_heads // v.shape[2]
            for i in range(max_num_heads):
                idx = i // block_size
                head_i = v[:, :, idx, :]
                item.append(
                    torch.mean(
                        head_i.reshape(v.shape[0] * v.shape[1], -1),
                        dim=0,
                        keepdim=False,
                    )
                    .to(dtype=torch.float32)
                    .cpu()
                )
            f_states.append(torch.stack(item))
        return torch.stack(f_states)

    return flatten_dict(query_states), flatten_dict(key_states)


def get_fig_ax():
    r, c = 4, 3
    fig, axes = plt.subplots(r, c, figsize=(c * 4, r * 2))
    axes = axes.flatten()
    return fig, axes


def visualize(query_states, key_states):
    dir = "../images/2-norm"
    num_layers = query_states.shape[0]
    # query
    for idx in range(0, num_layers, 12):
        st, ed = idx, min(idx + 12, num_layers)
        fig, axes = get_fig_ax()
        for i in range(0, 12):
            if i + st >= ed:
                break
            # 获取数据
            data = query_states[i + st].numpy()
            sns.heatmap(
                data,
                ax=axes[i],
                cmap="Greens",
                xticklabels=False,  # 关闭默认横轴标签
                yticklabels=False,  # 关闭默认纵轴标签
            )
            axes[i].set_title(f"Layer {i + 1}")

            # 自定义横轴刻度标签，显示每隔一定间隔的刻度
            xticks = range(
                0, data.shape[1], max(1, data.shape[1] // 6)
            )  # 每隔若干步显示一个
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels([str(x) for x in xticks], rotation=45)

            # 自定义纵轴刻度标签，使其从上到下递减
            yticks = range(data.shape[0])
            axes[i].set_yticks(yticks)
            axes[i].set_yticklabels([str(y) for y in reversed(yticks)])  # 倒序显示

        fig.suptitle("Query x:n_dim y:head")
        fig.tight_layout()  # 自动调整子图的间距
        fig.savefig(f"{dir}/query_layer{st}_{ed}.png")

    # key
    for idx in range(0, num_layers, 12):
        st, ed = idx, min(idx + 12, num_layers)
        fig, axes = get_fig_ax()
        for i in range(0, 12):
            if i + st >= ed:
                break
            # 获取数据
            data = key_states[i + st].numpy()
            sns.heatmap(
                data,
                ax=axes[i],
                cmap="Greens",
                xticklabels=False,  # 关闭默认横轴标签
                yticklabels=False,  # 关闭默认纵轴标签
            )
            axes[i].set_title(f"Layer {i + 1}")

            # 自定义横轴刻度标签，显示每隔一定间隔的刻度
            xticks = range(
                0, data.shape[1], max(1, data.shape[1] // 6)
            )  # 每隔若干步显示一个
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels([str(x) for x in xticks], rotation=45)

            # 自定义纵轴刻度标签，使其从上到下递减
            yticks = range(data.shape[0])
            axes[i].set_yticks(yticks)
            axes[i].set_yticklabels([str(y) for y in reversed(yticks)])  # 倒序显示

        fig.suptitle("Key x:n_dim y:head")
        fig.tight_layout()  # 自动调整子图的间距
        fig.savefig(f"{dir}/key_layer{st}_{ed}.png")


def main():
    args = get_args()
    config_file = args.config_file
    trainer = DistributedTrainer(config_file)
    model = trainer.model
    dataloader = get_dataloader(trainer)
    dataloader = list(dataloader.values())[0]
    model.eval()
    model.to("cuda")
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

    for name, module in model.named_modules():
        module.to("cuda")
        if not isinstance(module, FlashRotaryEmbedding):
            continue
        module._forward = module.forward
        module.forward = MethodType(get_rotary_forward(module_name=name), module)
    num = args.sample_size
    bsz = None
    query_states = []
    key_states = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            if bsz is None:
                bsz = batch["input_ids"].shape[0]
            model(**batch)
            query, key = flatten_avg_query_key_states(
                query_states_dict, key_states_dict
            )
            query_states.append(query)
            key_states.append(key)
            query_states_dict.clear()
            key_states_dict.clear()
            num -= bsz
            if num == 0:
                break
    # hidden_states:[bsz, seqlen, num_heads, head_dim] 8 2048 9 64

    query_states = torch.stack(query_states)
    key_states = torch.stack(key_states)
    query_states = torch.mean(
        query_states, dim=0, keepdim=False
    )  # [n_layer][n_head][n_dim/2]
    key_states = torch.mean(key_states, dim=0, keepdim=False)
    # visualize(query_states, key_states)
    qk_states = query_states + key_states
    if qk_states.shape[1] != model.config.num_key_value_heads:
        layer_num, _, dim = query_states.shape
        qk_states = qk_states.view(
            layer_num, model.config.num_key_value_heads, -1, dim
        ).sum(dim=2)
    with open(os.path.join(args.output_dir, "qk_tensor.pth"), "wb") as f:
        torch.save(qk_states, f)


def partial_rope(query_states, key_states, index, rotary_embedding, position_ids):
    # query_states:[bsz, seqlen, num_heads, head_dim]
    # key_states:[bsz, seqlen, num_heads, head_dim]
    # index:[num_heads, head_dim // 2]
    cos, sin = rotary_embedding(query_states, position_ids)
    q_embed, k_embed = rotary_embedding.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    mask = torch.zeros(
        (k_embed.size(2), k_embed.size(3) // 2), dtype=torch.bool, device=q_embed.device
    )
    mask.scatter_(1, index, 1)
    mask_for_k = (
        torch.cat((mask, mask), dim=1).unsqueeze(0).unsqueeze(1).to(q_embed.device)
    )
    mask_for_q = torch.repeat_interleave(
        input=mask_for_k, repeats=int(query_states.size(2) // key_states.size(2)), dim=2
    ).to(q_embed.device)
    q_embed = torch.where(mask_for_q == 1, q_embed, query_states)
    k_embed = torch.where(mask_for_k == 1, k_embed, key_states)
    return q_embed, k_embed

def calculate_attn_score(query_states, key_states, index, rotary_embedding, position_ids):
    # query_states:[bsz, seqlen, num_heads, head_dim]
    # key_states:[bsz, seqlen, num_heads, head_dim]
    new_query_states, new_key_states = partial_rope(query_states, key_states, index, rotary_embedding,position_ids)
    new_query_states = new_query_states.transpose(1, 2)
    new_key_states = new_key_states.transpose(1, 2)
    from transformers.models.llama.modeling_llama import repeat_kv
    new_key_states = repeat_kv(new_key_states, new_query_states.size(1)//new_key_states.size(1))
    attn_score = torch.einsum("bhqd,bhkd->bhqk", new_query_states, new_key_states)
    attn_score = attn_score / (query_states.size(-1) ** 0.5)
    attn_weights = torch.softmax(attn_score, dim=-1)
    return attn_weights

def rope_lite(query_states,key_states,rotary_embedding,position_ids):
    query_states = query_states.squeeze()
    key_states = key_states.squeeze()
    bsz, seqlen, num_attention_heads, head_dim = query_states.shape
    _ , _ , num_key_value_heads , _ = key_states.shape
    index_all = torch.arange(0, head_dim // 2, device=query_states.device).repeat(num_key_value_heads,1)
    index_r = torch.arange(0, 0, device=query_states.device).repeat(num_key_value_heads,1)
    attn_base = calculate_attn_score(query_states, key_states, index_all, rotary_embedding, position_ids)
    gqa_group_size = query_states.size(2) // key_states.size(2)
    for i in range(head_dim//2):
        index_remaing = torch.full((index_all.size(0),index_all.size(1)-index_r.size(1)), dtype=torch.int64, fill_value=0)
        for j in range(index_remaing.shape[0]):
            mask = ~torch.isin(index_all[j], index_r[j])
            index_remaing[j,:] = index_all[j][mask]
        print(index_r.shape)
        print(index_remaing.shape)
        index_remaing = index_remaing.to(device=query_states.device)
        distance = torch.arange(0,0,device=query_states.device).repeat(num_key_value_heads,1)
        for j in range(index_remaing.shape[1]):
            index = torch.cat((index_r,index_remaing[:,j].unsqueeze(1)),dim=1)
            attn_new = calculate_attn_score(query_states, key_states, index, rotary_embedding, position_ids)
            l1_distance = torch.abs(attn_base - attn_new).mean(dim=(0,2,3))
            l1_distance = l1_distance.reshape(-1,gqa_group_size).mean(dim=1,keepdim=False).unsqueeze(1)
            distance = torch.cat((distance,l1_distance),dim=1)
        # print(distance.shape)
        min_index = torch.argmin(distance,dim=1).reshape(-1,1)
        index_r = torch.cat((index_r,torch.gather(index_remaing, 1, min_index)),dim=1)
    return index_r

def main_rope_lite():
    args = get_args()
    config_file = args.config_file
    trainer = DistributedTrainer(config_file)
    model = trainer.model
    dataloader = get_dataloader(trainer)
    dataloader = list(dataloader.values())[0]
    model.eval()
    model.to("cuda")
    from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

    for name, module in model.named_modules():
        module.to("cuda")
        if not isinstance(module, FlashRotaryEmbedding):
            continue
        module._forward = module.forward
        module.forward = MethodType(get_rotary_forward(module_name=name), module)
    bsz = None
    r = 4
    query_states = []
    key_states = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to("cuda") for key, value in batch.items()}
            assert "input_mask" in batch
            assert torch.all(batch["input_mask"] == 1)
            model(**batch)
            position_ids = torch.cumsum(batch["input_mask"], dim=-1, dtype=torch.int32) - 1
            if bsz is None:
                bsz = batch["input_ids"].shape[0]
                print(bsz)
            break
    i=0
    qk_tensor = []
    print("start search")
    with torch.no_grad():
        for (_,query_states),(_,key_states) in zip(query_states_dict.items(),key_states_dict.items()):
            query_states = query_states.to("cuda").squeeze()
            key_states = key_states.to("cuda").squeeze()
            rotary_embedding = model.model.decoder[i].pp_block.attn.rotary_embedding
            i=i+1
            index_r = rope_lite(query_states, key_states, rotary_embedding, position_ids)[:,:r]
            t = torch.zeros((key_states.size(2), key_states.size(3) // 2), dtype=torch.bfloat16, device=key_states.device).scatter_(1, index_r, 1)
            qk_tensor.append(t)
            print(f"finish search {i} layer")

    qk_tensor = torch.stack(qk_tensor)
    with open(os.path.join(args.output_dir, "qk_tensor.pth"), "wb") as f:
        torch.save(qk_tensor, f)


if __name__ == "__main__":
    main()
    # main_rope_lite()
