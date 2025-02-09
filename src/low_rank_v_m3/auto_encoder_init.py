import torch
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM,PreTrainedModel
from nanotron.data.nanoset import Nanoset
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
import os
from typing import Dict, List, Tuple, Union
import numpy as np
from nanotron.logging import warn_once
import logging
import yaml
import torch


from ..optim.optimizer import load_optimizer_scheduler

TYPE_DICT = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

class AttnForTraing(PreTrainedModel):
    config_class = LlamaConfig
    def __init__(self,config):
        super().__init__(config)
        from .patch_func_hf import CustomLlamaSdpaAttention,CustomLlamaAttention
        self.config = config
        self.model = torch.nn.ModuleList(
            [
                CustomLlamaSdpaAttention(
                    config=config,
                    layer_idx=layer_idx,
                ) 
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

    def post_init(self,original_model):
        self.original_model = original_model
        import os
        if os.getenv("AE_LOSS", "L2") == "L2":
            self.loss_func = torch.nn.MSELoss(reduction="sum")
        elif os.getenv("AE_LOSS") == "L1":
            self.loss_func = torch.nn.SmoothL1Loss(reduction="sum")
        else:
            raise ValueError("Unsupported loss function specified in AE_LOSS environment variable.")
        for layer_idx, layer in enumerate(self.original_model.model.layers):
            original_attn = layer.self_attn
            # q,k,v,o
            q_proj, k_proj, v_proj = original_attn.q_proj.weight.detach(), original_attn.k_proj.weight.detach(), original_attn.v_proj.weight.detach()
            self.model[layer_idx].q_proj.weight.data[:] = q_proj
            self.model[layer_idx].k_proj.weight.data[:] = k_proj
            self.model[layer_idx].v_proj.weight.data[:] = v_proj
            self.model[layer_idx].o_proj.weight.data[:] = original_attn.o_proj.weight.detach()
            # W_down_v, W_up_v
            U,S,V = torch.svd(torch.eye(original_attn.head_dim).to(dtype=torch.float32))
            # U,S,V = torch.svd(torch.eye(original_attn.num_key_value_heads * original_attn.head_dim).to(dtype=torch.float32))
            low_rank = self.config.SVD["low_rank"]
            # low_rank = self.config.SVD["low_rank"] * self.config.num_key_value_heads
            dtype = self.model[layer_idx].W_down_v.weight.dtype
            U = U[:,:low_rank].to(dtype=dtype)
            S = S[:low_rank].to(dtype=dtype)
            V = V[:,:low_rank].to(dtype=dtype)
            self.model[layer_idx].W_down_v.weight.data[:] = U.t()
            self.model[layer_idx].W_up_v.weight.data[:] = (torch.diag(S) @ V.t()).t()
            self.model[layer_idx].W_down_v.weight.requires_grad_(True)
            self.model[layer_idx].W_up_v.weight.requires_grad_(True)
        for name,named_param in self.original_model.named_parameters():
            named_param.requires_grad = False
        for name,named_param in self.model.named_parameters():
            if all([x not in name for x in ["W_down_v","W_up_v"]]):
                named_param.requires_grad = False
            else:
                named_param.requires_grad = True

        self.inputs = {}
        for layer_id, layer in enumerate(self.original_model.model.layers):
            attn = layer.self_attn
            original_forward = attn.forward
            
            def make_new_forward(layer_id, inp_dict):
                def new_forward(self,*args, **kwargs):
                    output = self.original_forward(*args, **kwargs)
                    inp_dict[layer_id] = (args, kwargs)
                    return output
                return new_forward
    
            import types
            attn.original_forward = original_forward
            attn.forward = types.MethodType(make_new_forward(layer_id, self.inputs), attn)

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],
        attention_mask: Union[torch.Tensor, TensorPointer],
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        self.inputs.clear()
        sharded_logits = self.original_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        loss = torch.zeros(1, device=input_ids.device)
        warn_once(
            msg = "Using MSELoss for RoPE loss",
            logger = logging.getLogger(__name__),
        )
        for layer_idx,layer in enumerate(self.original_model.model.layers):
            original_attn = layer.self_attn
            hidden_states = self.inputs[layer_idx][1]["hidden_states"]
            bsz, seq_len, _ = hidden_states.shape
            original_value_states = original_attn.v_proj(hidden_states).reshape(bsz,seq_len,original_attn.num_key_value_heads,original_attn.head_dim)
            value_states = self.model[layer_idx].get_value_states(
                *self.inputs[layer_idx][0],
                **self.inputs[layer_idx][1],
            ).transpose(1,2).view(*original_value_states.shape)
            layer_loss = self.loss_func(original_value_states, value_states)
            loss += layer_loss / (value_states.shape[0] * value_states.shape[1]* value_states.shape[2])
        loss = loss / len(self.original_model.model.layers)
        return {"loss": loss}


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_tokenizer_and_model(model_arguments, model_config):
    """Load tokenizer and model from configuration."""
    # model
    dtype = TYPE_DICT[model_arguments["dtype"]]
    model_name_or_path = model_arguments["model_name_or_path"]
    model_config = LlamaConfig(**model_config)
    if model_name_or_path is not None:
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    else:
        model = LlamaForCausalLM(model_config)
    # tokenizer
    tokenizer_name_or_path = model_arguments["tokenizer_name_or_path"]
    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # Warning
    original_model = model
    model = AttnForTraing(model_config)
    model.post_init(original_model)
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

    # Trainer
    model, tokenizer = load_tokenizer_and_model(
        config["ModelArguments"], config["ModelConfig"]
    )
    train_dataset = load_dataset(config, tokenizer)
    resume_from_checkpoint = config["TrainingArguments"]["resume_from_checkpoint"]
    training_args = TrainingArguments(**config["TrainingArguments"])
    # optimizer, lr_scheduler = load_optimizer_scheduler(model, config)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        # optimizers=(optimizer, lr_scheduler),
    )
    # train
    if resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint)
    else:
        trainer.train()

def merge():
    from .patch_func_hf import low_rank_patch_hf
    low_rank_patch_hf()
    original_model = AutoModelForCausalLM.from_pretrained("/home/binguo/data/MLA-FT/checkpoints/rope_v0_svd_v_method2_rank8_silu/0_hf")
    model = AttnForTraing.from_pretrained("/home/binguo/data/MLA-FT/checkpoints/rope_v0_svd_v_method2_rank8_silu_auto_encoder_l1/checkpoint-2000")
    with torch.no_grad():
        for layer_idx, layer in enumerate(original_model.model.layers):
            attn = model.model[layer_idx]
            layer.self_attn.W_down_v.weight.data[:] = attn.W_down_v.weight.detach()
            layer.self_attn.W_up_v.weight.data[:] = attn.W_up_v.weight.detach()
    original_model.save_pretrained("/home/binguo/data/MLA-FT/checkpoints/rope_v0_svd_v_method2_rank8_silu_auto_encoder_l1/0_hf")

if __name__ == "__main__":
    main()
    # merge()
