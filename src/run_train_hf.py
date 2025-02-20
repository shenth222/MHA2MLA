import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from nanotron.data.nanoset import Nanoset
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import yaml


from .optim.optimizer import load_optimizer_scheduler

TYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_tokenizer_and_model(model_arguments, model_config):
    """Load tokenizer and model from configuration."""
    # model
    dtype = TYPE_DICT[model_arguments["dtype"]]
    model_name_or_path = model_arguments["model_name_or_path"]
    if model_name_or_path is not None:
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    else:
        model_config = LlamaConfig(**model_config)
        model = LlamaForCausalLM(model_config)
    # tokenizer
    tokenizer_name_or_path = model_arguments["tokenizer_name_or_path"]
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # Warning
    return model, tokenizer


class CustomNanoset(Nanoset):
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Returns sequence_length + 1 tokens from the memmap dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, torch.LongTensor]: The input ids wrapped in a dictionary
        """
        item = super().__getitem__(idx)
        item["labels"] = item["input_ids"]
        item["attention_mask"] = torch.ones_like(item["input_ids"]).bool()
        return item


def load_dataset(config, tokenizer):
    """Load dataset from configuration."""
    data_arguments = config["DataArguments"]
    dataset_folders = data_arguments["dataset_folders"]
    dataset_weights = data_arguments["dataset_weights"]
    sequence_length = data_arguments["sequence_length"]
    traingingargs = TrainingArguments(**config["TrainingArguments"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    token_size = 4 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else 2
    global_batch_size = (
        traingingargs.per_device_train_batch_size
        * world_size
        * traingingargs.gradient_accumulation_steps
    )
    dataset = CustomNanoset(
        dataset_folders=dataset_folders,
        sequence_length=sequence_length,
        dataset_weights=dataset_weights,
        token_size=token_size,
        train_split_num_samples=global_batch_size * traingingargs.max_steps,
    )
    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    config = load_config(args.config_file)
    assert config["DataArguments"]["DP"] == int(os.environ.get("WORLD_SIZE", 1)), "DP is not equal to WORLD_SIZE"

    # Monkey Pacth for RoPE
    cfg_RoPE = config["RoPE"]
    assert cfg_RoPE is not None, "RoPE config is missing"
    from .patch_func_hf import create_custom_apply_rotary_pos_emb_hf
    from transformers.models.llama import modeling_llama

    modeling_llama.apply_rotary_pos_emb = create_custom_apply_rotary_pos_emb_hf(cfg_RoPE)
    if cfg_RoPE["partial_rope_version"] == 4:
        from .patch_func_hf import custom_forward_LlamaAttention,custom_forward_LlamaFlashAttention2,custom_forward_LlamaSdpaAttention
        modeling_llama.LlamaAttention.forward = custom_forward_LlamaAttention
        modeling_llama.LlamaFlashAttention2.forward = custom_forward_LlamaFlashAttention2
        modeling_llama.LlamaSdpaAttention.forward = custom_forward_LlamaSdpaAttention


    # Trainer
    model, tokenizer = load_tokenizer_and_model(
        config["ModelArguments"], config["ModelConfig"]
    )
    train_dataset = load_dataset(config, tokenizer)
    resume_from_checkpoint = config["TrainingArguments"]["resume_from_checkpoint"]
    training_args = TrainingArguments(**config["TrainingArguments"])
    optimizer, lr_scheduler = load_optimizer_scheduler(model, config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
    )
    # train
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint)
    else:
        if int(os.getenv("LOCAL_RANK",0))==0:
            trainer._save_checkpoint()
        trainer.train()


if __name__ == "__main__":
    main()
