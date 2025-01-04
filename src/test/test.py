"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""

import argparse
import yaml
from typing import Dict, cast
from types import MethodType
import seaborn as sns
import torch
import inspect

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
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

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
        for k, v in states.items():
            item = []
            v = v.squeeze()
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
            k = k.split(".")[2]
            f_states.append(torch.stack(item))
        return torch.stack(f_states)

    return flatten_dict(query_states), flatten_dict(key_states)


def old_visualize(query_states, key_states):
    dir = "/home/binguo/data/MLA-FN/iamges/2-norm"
    for k1, v in key_states.items():
        for k2, t in v.items():
            print(t.shape)
            t = t.to(torch.float32).detach().cpu().numpy()
            plt.bar(range(1, len(t) + 1), t)
            plt.xlabel("d_dim")  # x 轴标签
            plt.ylabel("2-norm")  # y 轴标签
            plt.savefig(f"{dir}/key_layer{k1}_head{k2}.png")
            plt.xticks(range(1, t.size + 1))  # 从1开始递增，假设 bins=30
            plt.close()

    for k1, v in query_states.items():
        for k2, t in v.items():
            t = t.to(torch.float32).detach().cpu().numpy()
            plt.bar(range(1, len(t) + 1), t)
            plt.xlabel("d_dim")
            plt.ylabel("2-norm")
            plt.savefig(f"{dir}/query_layer{k1}_head{k2}.png")
            plt.xticks(range(1, t.size + 1))  # 从1开始递增，假设 bins=30
            plt.close()


def get_fig_ax():
    r, c = 4, 3
    fig, axes = plt.subplots(r, c, figsize=(c * 4, r * 2))
    axes = axes.flatten()
    return fig, axes


def visualize(query_states, key_states):
    dir = "/home/binguo/data/MLA-FN/iamges/2-norm"
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




hidden_states_dict_nanotron = {}
output_states_dict_nanotron = {}

def get_custom_forward_nanotron(module_name):
    module_name = module_name.split(".")[2]
    def custom_forward(self, *args, **kwargs):
        global hidden_states_dict_nanotron
        hidden_states_dict_nanotron[module_name] = {}
        
        # 获取 _forward 的参数名
        param_names = list(inspect.signature(self._forward).parameters.keys())

        # 映射 *args 到对应的参数名
        args_dict = dict(zip(param_names, args))

        # 合并 args 和 kwargs 到 hidden_states_dict_nanotron[module_name]
        for param_name in param_names:
            if param_name in args_dict:
                hidden_states_dict_nanotron[module_name][param_name] = args_dict[param_name]
            elif param_name in kwargs:
                hidden_states_dict_nanotron[module_name][param_name] = kwargs[param_name]
            else:
                hidden_states_dict_nanotron[module_name][param_name] = None  # 如果参数没有提供，设置为 None

        ouput= self._forward(*args, **kwargs)
        output_states_dict_nanotron[module_name] = ouput
        return ouput
    return custom_forward


hidden_states_dict_hf = {}
output_states_dict_hf = {}

def get_custom_forward_hf(module_name):
    module_name = module_name.split(".")[2]
    def custom_forward(self, *args, **kwargs):
        global hidden_states_dict_hf
        hidden_states_dict_hf[module_name] = {}
        
        # 获取 _forward 的参数名
        param_names = list(inspect.signature(self._forward).parameters.keys())

        # 映射 *args 到对应的参数名
        args_dict = dict(zip(param_names, args))

        # 合并 args 和 kwargs 到 hidden_states_dict_hf[module_name]
        for param_name in param_names:
            if param_name in args_dict:
                hidden_states_dict_hf[module_name][param_name] = args_dict[param_name]
            elif param_name in kwargs:
                hidden_states_dict_hf[module_name][param_name] = kwargs[param_name]
            else:
                hidden_states_dict_hf[module_name][param_name] = None  # 如果参数没有提供，设置为 None

        output=self._forward(*args, **kwargs)
        output_states_dict_hf[module_name] = output
        return output

    return custom_forward


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    # model_nanotron
    trainer = DistributedTrainer(config_file)
    model_nanotron = trainer.model
    dataloader = get_dataloader(trainer)
    dataloader = list(dataloader.values())[0]
    model_nanotron.eval()
    model_nanotron.to("cuda")
    suffix = "attn"
    num_nanotron = sum(p.numel() for p in model_nanotron.parameters())
    for name, module in model_nanotron.named_modules():
        if not name.endswith(suffix):
            continue
        module._forward = module.forward
        module.forward = MethodType(get_custom_forward_nanotron(name), module)

    # model_hf
    model_hf = AutoModelForCausalLM.from_pretrained(
        "/home/binguo/data/models/HuggingFaceTB/SmolLM-135M"
    )
    num_hf = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    model_hf.to("cuda", dtype=torch.bfloat16)
    for name, module in model_hf.named_modules():
        if not name.endswith(suffix):
            continue
        module._forward = module.forward
        module.forward = MethodType(get_custom_forward_hf(name), module)
    print(num_nanotron, num_hf)

    num = 1
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/binguo/data/models/HuggingFaceTB/SmolLM-135M"
    )
    # print(model_nanotron)
    # print(model_hf)
    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to("cuda") for key, value in batch.items()} # dict_keys(['input_ids', 'input_mask', 'label_ids', 'label_mask'])
            print(model_nanotron(**batch))
            print(model_hf(input_ids=batch["input_ids"],labels=batch["input_ids"],attention_mask=batch["input_mask"]).loss)
            num -= 1
            if num == 0:
                break
    # compare hidden_states
    # for i in range(30):
    #     print(torch.allclose(hidden_states_dict_nanotron[str(i)]["hidden_states"].transpose(0,1),hidden_states_dict_hf[str(i)]["hidden_states"],atol=1e-3))
    #     print(output_states_dict_nanotron[str(i)]["hidden_states"].shape)
    #     print(output_states_dict_hf[str(i)][0].shape)
    #     print(torch.allclose(output_states_dict_nanotron[str(i)]["hidden_states"].transpose(0,1),output_states_dict_hf[str(i)][0],atol=1e-3))
        # if torch.allclose(hidden_states_dict_nanotron[str(i)]["hidden_states"].transpose(0,1),hidden_states_dict_hf[str(i)]["hidden_states"],atol=1e-3):
        #     print(output_states_dict_nanotron[str(i)].shape)
        #     print(output_states_dict_hf[str(i)].shape)
        #     print(torch.allclose(output_states_dict_nanotron[str(i)].transpose(0,1),output_states_dict_hf[str(i)],atol=1e-3))



    # input_ids = torch.randint(0, tokenizer.vocab_size, (1, 1)).to("cuda")
    # input_mask = torch.ones(1, 1).to("cuda").bool()
    # print(input_ids)
    # model_nanotron.model.token_position_embeddings.pp_block.token_embedding.weight = model_hf.model.embed_tokens.weight
    # embedding_nanotron = model_nanotron.model.token_position_embeddings(
    #     input_ids=input_ids, input_mask=input_mask
    # )["input_embeds"]
    # embedding_hf = model_hf.model.embed_tokens(input_ids)
    # print(embedding_nanotron)
    # print(embedding_hf)
    # print(torch.allclose(embedding_nanotron, embedding_hf, atol=1e-3))
