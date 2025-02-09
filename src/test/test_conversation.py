"""
Nanotron training script.

Usage:
```
export CUDA_DEVICE_MAX_CONNECTIONS=1 # important for some distributed operations
torchrun --nproc_per_node=8 run_train.py --config-file examples/config_tiny_llama.yaml
```
"""

import torch
from transformers import LlamaConfig as LlamaConfig_hf
from nanotron.config import LlamaConfig as LlamaConfig_nt
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv


import argparse
import os
import math
from pathlib import Path
from torch import nn

import torch
from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    GenerationArgs,
    LoggingArgs,
    ParallelismArgs,
    get_config_from_file,
)
from nanotron.generation.decode import (
    GenerationInput,
    TokenizerConfig,
    decode_text,
    decode_tokenized,
)
from nanotron.logging import log_rank, set_ranks_logging_level
from nanotron.models import build_model
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    OneForwardOneBackwardPipelineEngine,
)
from nanotron.parallel.pipeline_parallel.tensor_pointer import TensorPointer
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.random import (
    RandomStates,
    get_current_random_state,
    get_synced_random_state,
    set_random_seed,
)
from nanotron.serialize import load_weights
from nanotron.trainer import CONFIG_TO_MODEL_CLASS, mark_tied_parameters

from ..mla.mla_patch_hf import (
    CustomLlamaSdpaAttention,
    mla_patch_hf,
    CustomLlamaAttention,
    CustomLlamaFlashAttention2,
)
from ..mla.mla_patch_nt import CustomCausalSelfAttention, mla_patch_nt


from transformers.cache_utils import Cache, StaticCache
import argparse
import torch
import yaml
from typing import Dict, Optional, Tuple, cast

import numpy as np
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
    from transformers import AutoTokenizer, AutoModelForCausalLM
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


def compare_instances(instance1, instance2):
    """
    比较两个实例的属性，并打印出不同的属性。
    """
    # 获取两个实例的所有属性
    attrs1 = set(dir(instance1))
    attrs2 = set(dir(instance2))

    # 找到两个实例共有的属性
    common_attrs = attrs1.intersection(attrs2)

    # 遍历共有属性，比较值
    for attr in common_attrs:
        if not attr.startswith("__"):  # 过滤掉内置属性
            value1 = getattr(instance1, attr)
            value2 = getattr(instance2, attr)

            # 如果值不同，则打印
            if value1 != value2:
                print(f"属性 '{attr}' 不同:")
                print(f"  实例1: {value1}")
                print(f"  实例2: {value2}")
                print("-" * 40)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the YAML or python config file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file
    torch.set_default_dtype(torch.bfloat16)

    # Monkey patch
    from ..mla.mla_patch_nt import (
        mla_patch_nt,
        CustomModelArgs,
        CustomLlamaConfig,
        CustomConfig,
    )
    from ..mla.mla_patch_hf import mla_patch_hf
    import yaml

    with open(config_file, "r") as fin:
        config = yaml.safe_load(fin)
    rope_cfg = config["model"]["model_config"]["RoPE"]
    mla_patch_nt(rope_cfg)
    mla_patch_hf(rope_cfg)

    from nanotron import trainer as nt_trainer

    # Load trainer and data
    trainer = nt_trainer.DistributedTrainer(config_file, config_class=CustomConfig)
    model = trainer.model
    # print(trainer.unwrapped_model.config.rope_interleaved)
    dataloader = get_dataloader(trainer)
    model_nanotron = trainer.model
    dataloader = list(dataloader.values())[0]
    model_hf = AutoModelForCausalLM.from_pretrained(
        # "../checkpoints/rope_v4_topk4_svd_method7_rank8/18000_hf"
        "/home/binguo/data/MLA-FT/checkpoints/test_hf"
    ).cuda()
    num = 1
    with torch.no_grad():
        model_nanotron.eval()
        for batch in dataloader:
            batch = {key: value.to("cuda") for key, value in batch.items()} # dict_keys(['input_ids', 'input_mask', 'label_ids', 'label_mask'])
            input_ids = batch["input_ids"]
            input_mask = batch["input_mask"]
            print(model_nanotron(**batch))
            print(model_hf(input_ids=batch["input_ids"],labels=batch["input_ids"],attention_mask=batch["input_mask"]).loss)
            print(sum(p.numel() for p in model_hf.parameters()))
            print(model_nanotron)
            print(model_hf)
            num -= 1
            if num == 0:
                break

    state_dict_hf = model_hf.state_dict()
    state_dict_nanotron = model_nanotron.state_dict()
    for i in range(30):
        hf_prefix = f"model.layers.{i}"
        nt_prefix = f"model.decoder.{i}.pp_block"
        hf_to_nt_map = {}
        stata_dict_layer1_hf = {}
        hf_to_nt_map[f"{hf_prefix}.self_attn.q_proj.weight"] = (
            f"{nt_prefix}.attn.q_proj.weight"
        )
        hf_to_nt_map[f"{hf_prefix}.self_attn.W_k_r.weight"] = (
            f"{nt_prefix}.attn.W_k_r.weight"
        )
        hf_to_nt_map[f"{hf_prefix}.self_attn.W_down_k.weight"] = (
            f"{nt_prefix}.attn.W_down_k.weight"
        )
        hf_to_nt_map[f"{hf_prefix}.self_attn.W_down_v.weight"] = (
            f"{nt_prefix}.attn.W_down_v.weight"
        )
        hf_to_nt_map[f"{hf_prefix}.self_attn.W_up_k.weight"] = (
            f"{nt_prefix}.attn.W_up_k.weight"
        )
        hf_to_nt_map[f"{hf_prefix}.self_attn.W_up_v.weight"] = (
            f"{nt_prefix}.attn.W_up_v.weight"
        )
        for key, value in hf_to_nt_map.items():
            if key not in state_dict_hf:
                continue
            assert torch.equal(
                state_dict_nanotron[value], state_dict_hf[key]
            ), f"{key} not equal {value}"
    assert torch.equal(
        state_dict_hf["lm_head.weight"],
        state_dict_nanotron["model.lm_head.pp_block.weight"],
    )
    assert torch.equal(
        state_dict_hf["model.embed_tokens.weight"],
        state_dict_nanotron[
            "model.token_position_embeddings.pp_block.token_embedding.weight"
        ],
    )
    assert torch.equal(
        state_dict_hf["model.norm.weight"],
        state_dict_nanotron["model.final_layer_norm.pp_block.weight"],
    )
